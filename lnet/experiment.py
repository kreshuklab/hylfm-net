import logging
import matplotlib.pyplot as plt
import numpy
import time
import torch.nn

from dataclasses import dataclass
from enum import Enum
from ignite.engine import Events, Engine
from ignite.handlers import ModelCheckpoint, EarlyStopping, TerminateOnNan
from ignite.metrics import Loss
from ignite.utils import convert_tensor
from importlib import import_module
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pathlib import Path
from scipy.special import expit
from typing import Union, Optional, List, Dict, Callable, Type, Any, Tuple, Sequence, Iterable, Generator

from inferno.io.transform import Transform
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, SubsetRandomSampler, ConcatDataset

from lnet.config import Config
from lnet.datasets import DatasetFactory, SubsetSequentialSampler, Result
from lnet.engine import TunedEngine, TrainEngine, EvalEngine
from lnet.output import Output
from lnet.step_functions import training_step, inference_step
from lnet.utils.metrics import (
    LOSS_NAME,
    AUX_LOSS_NAME,
    NRMSE_NAME,
    PSNR_NAME,
    SSIM_NAME,
    MSSSIM_NAME,
    BEAD_PRECISION_RECALL,
    BEAD_PRECISION,
    BEAD_RECALL,
)

from lnet.utils.plotting import turbo_colormap_data, Box, ColorSelection
from lnet.utils.transforms import lightfield_from_channel, EdgeCrop



class Experiment:
    train_dataset_factory: DatasetFactory
    valid_dataset_factory: Optional[DatasetFactory] = None
    test_dataset_factory: Optional[DatasetFactory] = None

    train_data_indices: List[Optional[List[int]]]
    train_eval_data_indices: List[Optional[List[int]]]
    valid_data_indices: List[Optional[List[int]]]
    test_data_indices: List[Optional[List[int]]]

    train_transforms: List[Union[Generator[Transform, None, None], Transform]]
    valid_transforms: List[Union[Generator[Transform, None, None], Transform]]
    test_transforms: List[Union[Generator[Transform, None, None], Transform]]

    train_loader: DataLoader
    train_loader_eval: DataLoader
    valid_loader: DataLoader
    test_loader: DataLoader

    max_num_epochs: int
    score_function: Callable[[Engine], float]

    config: Config
    add_in_name: Optional[str] = None

    def __init__(self, config_path: Path):
        self.config_path = config_path
        if config_path.suffix == ".py":
            config_module_name = (
                config_path.absolute()
                .relative_to(Path(__file__).parent.parent)
                .with_suffix("")
                .as_posix()
                .replace("/", ".")
            )
            config_module = import_module(config_module_name)
            config = getattr(config_module, "config")
            assert isinstance(config, Config)
        else:
            config = Config.from_yaml(config_path)

        self.config = config

        self.dtype = getattr(torch, config.model.precision)

        self.max_num_epochs = config.train.max_num_epochs
        self.score_function = config.train.score_function


        # # make sure everything is commited
        # git_status_cmd = pbs3.git.status("--porcelain")
        # git_status = git_status_cmd.stdout
        # assert not git_status, git_status  # uncommited changes exist

        self.pre_loss_transform: Callable[
            [torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]
        ] = lambda *x: x
        self.pre_aux_loss_transform: Callable[
            [torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]
        ] = lambda *x: x


        self.logger = logging.getLogger(config.log.time_stamp)

        z_out = None
        for data_config in [config.train_data, config.valid_data, config.test_data]:
            if data_config is None:
                continue

            if z_out is None:
                z_out = data_config.factory.get_z_out()
            else:
                assert z_out == data_config.factory.get_z_out(), (z_out, data_config.factory.get_z_out())

        assert z_out is not None
        self.z_out = z_out

        self.result_dir = self.config.log.dir / "result"

        devices = list(range(torch.cuda.device_count()))
        assert len(devices) == 1, "single gpu for now only"
        self.device = torch.device("cuda", devices[0])
        self.model = config.model.Model(nnum=config.model.nnum, z_out=self.z_out, **config.model.kwargs).to(
            device=self.device, dtype=self.dtype
        )
        self.model.cuda()
        if config.model.checkpoint is not None:
            state = torch.load(config.model.checkpoint, map_location=self.device)
            self.model.load_state_dict(state, strict=False)


    def test(self):
        self.max_num_epochs = 0
        self.run()

    def run(self):
        config = self.config
        # tensorboardX
        writer = SummaryWriter(self.config.log.dir.as_posix())
        # data_loader_iter = iter(train_loader)
        # x, y = next(data_loader_iter)
        # try:
        #     writer.add_graph(self.model, x.to(torch.device("cuda")))
        # except Exception as e:
        #     self.logger.warning("Failed to save model graph...")
        #     self.logger.exception(e)


        trainer = TrainEngine(process_function=training_step, config=config, logger=self.logger, model=self.model)
        return

        train_evaluator = EvalEngine(process_function=inference_step, config=config, logger=self.logger, model=self.model, data_config=config.train_eval_data)
        validator = EvalEngine(process_function=inference_step, config=config, logger=self.logger, model=self.model, data_config=config.valid_data)
        tester = EvalEngine(process_function=inference_step, config=config, logger=self.logger, model=self.model, data_config=config.test_data)

        saver = ModelCheckpoint(
            (self.config.log.dir / "models").as_posix(),
            "v0",
            score_function=self.score_function,
            n_saved=1,
            create_dir=True,
            save_as_state_dict=True,
        )
        validator.add_event_handler(
            Events.COMPLETED, saver, {"model": self.model}
        )  # , "optimizer": optimizer})

        stopper = EarlyStopping(
            patience=config.train.patience, score_function=self.score_function, trainer=trainer
        )
        validator.add_event_handler(Events.COMPLETED, stopper)

        Loss(loss_fn=lambda loss, _: loss, output_transform=lambda out: (out.loss, out.ipt)).attach(
            validator, LOSS_NAME
        )
        if len(self.loss_fn) > 1:
            for i in range(len(self.loss_fn)):
                Loss(loss_fn=lambda loss, _: loss, output_transform=lambda out, j=i: (out.losses[j], out.ipt)).attach(
                    validator, f"{LOSS_NAME}-{i}"
                )

        if self.aux_loss_fn is not None:
            Loss(loss_fn=lambda loss, _: loss, output_transform=lambda out: (out.aux_loss, out.ipt)).attach(
                validator, AUX_LOSS_NAME
            )
            for i in range(len(self.aux_loss_fn)):
                Loss(
                    loss_fn=lambda loss, _: loss, output_transform=lambda out, j=i: (out.aux_losses[j], out.ipt)
                ).attach(validator, f"{AUX_LOSS_NAME}-{i}")

        def log_train_scalars(engine: Engine, step: int):
            writer.add_scalar(f"{engine.state.name}/loss", engine.state.output.loss, step)

        def log_images(engine: Engine, step: int, boxes: Iterable[Box] = tuple()):
            output: Output = engine.state.output
            ipt_batch = numpy.stack([lightfield_from_channel(xx, nnum=engine.config.model.nnum) for xx in output.ipt.cpu().numpy()])

            tgt_batch = numpy.stack([yy.cpu().numpy() for yy in output.tgt])
            pred_batch = numpy.stack([yy.detach().cpu().numpy() for yy in output.pred])
            assert ipt_batch.shape[0] == tgt_batch.shape[0], (ipt_batch.shape, tgt_batch.shape)
            assert len(tgt_batch.shape) == 5, tgt_batch.shape
            assert tgt_batch.shape[1] == 1, tgt_batch.shape

            has_aux = output.aux_tgt is not None
            if has_aux:
                aux_tgt_batch = numpy.stack([yy.cpu().numpy() for yy in output.aux_tgt])
                aux_pred_batch = numpy.stack([yy.detach().cpu().numpy() for yy in output.aux_pred])
                aux_pred_batch = expit(aux_pred_batch)
                assert ipt_batch.shape[0] == aux_tgt_batch.shape[0], (ipt_batch.shape, aux_tgt_batch.shape)
                assert len(aux_tgt_batch.shape) == 5, aux_tgt_batch.shape
                assert aux_tgt_batch.shape[1] == 1, aux_tgt_batch.shape
            else:
                aux_tgt_batch = None
                aux_pred_batch = None

            fig, ax = plt.subplots(
                nrows=ipt_batch.shape[0],
                ncols=4 + 3 * int(has_aux),
                squeeze=False,
                figsize=(4 * 3, ipt_batch.shape[0] * 3),
            )
            fig.subplots_adjust(hspace=0, wspace=0, bottom=0, top=1, left=0, right=1)

            def make_subplot(ax, img, boxes: Iterable[Box] = tuple()):
                im = ax.imshow(img, cmap=ListedColormap(turbo_colormap_data))
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.axis("off")
                for box in boxes:
                    box.apply_to_ax(ax)

                # from https://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
                # create an axes on the right side of ax. The width of cax will be 5%
                # of ax and the padding between cax and ax will be fixed at 0.05 inch.
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="3%", pad=0.03)
                fig.colorbar(im, cax=cax)
                # ax.set_title(f"min-{img.min():.2f}-max-{img.max():.2f}")  # taking too much space!

            ax[0, 0].set_title("input")
            ax[0, 1].set_title("target")
            ax[0, 2].set_title("prediction")
            ax[0, 3].set_title("rel diff")
            for i, (ib, tb, pb) in enumerate(zip(ipt_batch, tgt_batch, pred_batch)):
                make_subplot(ax[i, 0], ib[0])
                make_subplot(ax[i, 1], tb[0].max(axis=0), boxes=boxes)
                make_subplot(ax[i, 2], pb[0].max(axis=0), boxes=boxes)
                make_subplot(ax[i, 3], (numpy.abs(pb - tb) / tb)[0].max(axis=0))

            if has_aux:
                ax[0, 4].set_title("aux target")
                ax[0, 5].set_title("aux prediction")
                ax[0, 6].set_title("abs diff")
                for i, (atb, apb) in enumerate(zip(aux_tgt_batch, aux_pred_batch)):
                    make_subplot(ax[i, 4], atb[0].max(axis=0), boxes=boxes)
                    make_subplot(ax[i, 5], apb[0].max(axis=0), boxes=boxes)
                    make_subplot(ax[i, 6], numpy.abs(apb - atb)[0].max(axis=0))

            fig.tight_layout()

            writer.add_figure(f"{engine.state.name}/in_out", fig, step)
            # ipt_batch = numpy.concatenate([ipt_batch] * 3, axis=1)  # make channel dim = 3 (NHW gives error)
            # writer.add_images(f"in_training/input", ipt_batch, step, dataformats="NCHW")

        @validator.on(Events.COMPLETED)
        def log_eval(engine: TunedEngine):
            met = validator.state.metrics
            self.logger.info("%s - Epoch: %d  Avg loss: %.3f", engine.state.name, engine.state.epoch, met[LOSS_NAME])

            available_metrics = [LOSS_NAME, MSSSIM_NAME, SSIM_NAME, PSNR_NAME, NRMSE_NAME]
            if len(self.loss_fn) > 1:
                for i in range(len(self.loss_fn)):
                    available_metrics.append(f"{LOSS_NAME}-{i}")

            if self.aux_loss_fn is not None:
                available_metrics.append(AUX_LOSS_NAME)
                for i in range(len(self.aux_loss_fn)):
                    available_metrics.append(f"{AUX_LOSS_NAME}-{i}")

            if BEAD_PRECISION_RECALL in met:
                bprecision, brecall = met[BEAD_PRECISION_RECALL]
                if not numpy.isnan(bprecision):
                    met[BEAD_PRECISION] = bprecision
                    available_metrics.append(BEAD_PRECISION)

                if not numpy.isnan(brecall):
                    met[BEAD_RECALL] = brecall
                    available_metrics.append(BEAD_RECALL)

            def log_metric(metric: str):
                writer.add_scalar(f"{engine.state.name}/{metric}", met[metric], trainer.state.epoch)
                metric_log_file = self.result_dir / engine.state.name / f"{metric.lower()}.txt"
                with metric_log_file.open(mode="a") as file:
                    file.write(f"{trainer.state.epoch}\t{met[metric]}\n")

            [log_metric(metric) for metric in available_metrics]

            boxes = []
            if engine.state.ipaths_log.shape[1]:
                fig, ax = plt.subplots()
                n_times = engine.state.ipaths_log.shape[0]
                # ax.set_xlim([0, n_times - 1])
                ax.set_ylim([0, 1.0])
                colors = ColorSelection(["b", "g", "r", "c", "m", "y"])

                # log path data to file as well
                this_path_dir = self.result_dir / engine.state.name / "paths" / f"epoch{trainer.state.epoch}"
                this_path_dir.mkdir(parents=True, exist_ok=True)

                for ip in range(engine.state.ipaths_log.shape[1]):
                    intensities = engine.state.ipaths_log[:, ip]
                    target_intensities = engine.state.target_ipaths_log[:, ip]
                    times = numpy.arange(n_times)
                    mask = numpy.greater_equal(intensities, 0)

                    # log to txt
                    with (this_path_dir / f"path{ip}.txt").open(mode="a") as file:
                        file.write(
                            "".join(
                                f"{time}\t{i_pred}\t{i_target}\n"
                                for time, i_pred, i_target in zip(
                                    times[mask], intensities[mask], target_intensities[mask]
                                )
                            )
                        )

                    # plot for tensorboardX
                    ax.plot(times[mask], intensities[mask], colors[ip], label=f"path {ip:2.0f}")
                    ax.plot(times[mask], target_intensities[mask], ":" + colors[ip], label=f"path {ip:2.0f} target")

                    # for timepoint in [0, -1]:
                    for timepoint in range(len(engine.state.ipaths)):
                        this_ipath = engine.state.ipaths[timepoint][ip]
                        if this_ipath is not None:
                            ipath_y, ipath_x = this_ipath[2:]
                            boxes.append(Box(ipath_x, ipath_y, colors[ip]))

                ax.legend()
                fig.tight_layout()
                writer.add_figure(f"{engine.state.name}/paths", fig, trainer.state.epoch)

            log_images(engine, trainer.state.epoch, boxes)

            # for i, yy in enumerate(experiment.state.output["y"]):
            #     yy = yy.cpu().numpy()
            #     assert len(yy.shape) == 4, yy.shape
            #     assert yy.shape[0] == 1, yy.shape
            #     writer.add_images(f"{name}/output{i}", yy.max(axis=1), step, dataformats="CHW")

        # desc = "EPOCH {} - loss: {:.3f}"
        # pbar = tqdm(initial=0, leave=False, total=len(train_loader), desc=desc.format(1, 0))

        test_result = Result(self.result_dir / "test" / "prediction", self.result_dir / "test" / "target")

        @validator.on(Events.ITERATION_COMPLETED)
        def eval_iteration(engine: Engine):
            output: Output = engine.state.output
            start = ((engine.state.iteration - 1) % len(engine.state.dataloader)) * self.config.eval.batch_size
            pred_batch = output.pred.detach().cpu().numpy()
            tgt_batch = output.tgt.detach().cpu().numpy()
            for in_batch_idx, ds_idx in enumerate(range(start, start + tgt_batch.shape[0])):
                if ds_idx >= len(engine.state.ipaths):
                    continue

                for ip, slices in enumerate(engine.state.ipaths[ds_idx]):
                    if slices is not None:
                        intensity = pred_batch[(in_batch_idx,) + slices].mean()
                        target_intensity = tgt_batch[(in_batch_idx,) + slices].mean()
                        self.logger.info(
                            "Measured output intensity %f (target=%f) at t=%d for path %d",
                            intensity,
                            target_intensity,
                            ds_idx,
                            ip,
                        )
                        engine.state.ipaths_log[ds_idx, ip] = intensity
                        engine.state.target_ipaths_log[ds_idx, ip] = target_intensity

            if engine.state.name == "test":
                test_result.update(pred_batch, tgt_batch, at=start)

        @trainer.on(Events.ITERATION_COMPLETED)
        def log_training_iteration(engine):
            iteration = engine.state.iteration
            it_in_epoch = (iteration - 1) % len(self.train_loader) + 1
            if (
                self.config.log.log_scalars_every[1] == Events.ITERATION_COMPLETED
                and it_in_epoch % self.config.log.log_scalars_every[0] == 0
            ):
                log_train_scalars(engine, iteration)

            if (
                self.config.log.log_images_every[1] == Events.ITERATION_COMPLETED
                and it_in_epoch % self.config.log.log_images_every[0] == 0
            ):
                log_images(engine, iteration)

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_training_epoch(engine):
            epoch = engine.state.epoch
            if (
                self.config.log.log_scalars_every[1] == Events.EPOCH_COMPLETED
                and epoch % self.config.log.log_scalars_every[0] == 0
            ):
                log_train_scalars(engine, epoch)

            if (
                self.config.log.log_images_every[1] == Events.EPOCH_COMPLETED
                and epoch % self.config.log.log_images_every[0] == 0
            ):
                log_images(engine, epoch)

        @trainer.on(Events.EPOCH_COMPLETED)
        def validate(engine):
            if engine.state.epoch % self.config.train.validate_every_nth_epoch == 0:
                # evaluate on training data
                validator.run(self.train_loader_eval)

                # evaluate on validation data
                if self.valid_loader is not None:
                    validator.run(self.valid_loader)

                validator.fire_event(CustomEvents.VALIDATION_DONE)

        if self.config.train.focus is not None:

            @trainer.on(Events.EPOCH_COMPLETED)
            def decay_focus_weight(engine):
                if engine.state.epoch % self.config.train.focus.decay_every_nth_epoch == 0:
                    engine.state.focus_weight = (
                        engine.state.focus_weight - 1.0
                    ) * self.config.train.focus.decay_factor + 1.0

        @trainer.on(Events.COMPLETED)
        def log_test_results(engine: TunedEngine):
            if self.test_loader is not None:
                if saver._saved:
                    score, file_list = saver._saved[-1]
                    self.model.load_state_dict(
                        torch.load(file_list[0], map_location=next(self.model.parameters()).device)
                    )

                validator.run(self.test_loader)

        trainer.run(self.train_loader, max_epochs=self.max_num_epochs)
        writer.close()
        if self.config_path is not None:
            with self.config_path.with_suffix(".ran_on.txt").open("a") as f:
                f.write(self.config.log.commit_hash)

    def export_test_predictions(self, output_path: Optional[str] = None):
        raise NotImplementedError
