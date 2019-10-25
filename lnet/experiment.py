import logging
import matplotlib.pyplot as plt
import numpy
import torch.nn

from ignite.engine import Events, Engine
from ignite.handlers import ModelCheckpoint, EarlyStopping
from importlib import import_module
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pathlib import Path
from scipy.special import expit
from tensorboardX import SummaryWriter
from typing import Union, Optional, Callable, Tuple, Iterable

from torch.utils.data import ConcatDataset

from lnet.config.config import Config
from lnet.datasets import Result
from lnet.engine import TunedEngine, TrainEngine, EvalEngine
from lnet.output import Output, AuxOutput
from lnet.step_functions import training_step, inference_step
from lnet.metrics import (
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

from lnet.utils.plotting import turbo_colormap_data, Box
from lnet.utils.transforms import lightfield_from_channel


class Experiment:
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
        for data_config in [
            config.train.data,
            config.eval_.eval_train_data,
            config.eval_.valid_data,
            config.eval_.test_data,
        ]:
            if data_config is None:
                continue

            if z_out is None:
                z_out = data_config.z_out
            else:
                assert z_out == data_config.z_out, (z_out, data_config.z_out)

        assert z_out is not None
        self.z_out = z_out

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

        trainer = (
            None
            if config.train is None
            else TrainEngine(process_function=training_step, config=config, logger=self.logger, model=self.model)
        )

        train_evaluator = (
            None
            if config.eval_.eval_train_data is None
            else EvalEngine(
                process_function=inference_step,
                config=config,
                logger=self.logger,
                model=self.model,
                data_config=config.eval_.eval_train_data,
            )
        )
        validator = (
            None
            if config.eval_.valid_data is None
            else EvalEngine(
                process_function=inference_step,
                config=config,
                logger=self.logger,
                model=self.model,
                data_config=config.eval_.valid_data,
            )
        )
        tester = (
            None
            if config.eval_.test_data is None
            else EvalEngine(
                process_function=inference_step,
                config=config,
                logger=self.logger,
                model=self.model,
                data_config=config.eval_.test_data,
            )
        )

        saver = ModelCheckpoint(
            (self.config.log.dir / "models").as_posix(),
            "v0",
            score_function=self.score_function,
            n_saved=config.log.save_n_checkpoints,
            create_dir=True,
            save_as_state_dict=True,
        )
        validator.add_event_handler(Events.COMPLETED, saver, {"model": self.model})  # , "optimizer": optimizer})

        stopper = EarlyStopping(patience=config.train.patience, score_function=self.score_function, trainer=trainer)
        validator.add_event_handler(Events.COMPLETED, stopper)

        def log_train_scalars(engine: TrainEngine, step: int):
            writer.add_scalar(f"{engine.name}/loss", engine.state.output.loss, step)

        def log_images(engine: TunedEngine, step: int, boxes: Iterable[Box] = tuple()):
            output: Union[AuxOutput, Output] = engine.state.output
            ipt_batch = numpy.stack(
                [lightfield_from_channel(xx, nnum=engine.config.model.nnum) for xx in output.ipt.cpu().numpy()]
            )

            # tgt_batch = numpy.stack([yy.cpu().numpy() for yy in output.tgt])
            # pred_batch = numpy.stack([yy.detach().cpu().numpy() for yy in output.pred])
            tgt_batch = output.tgt.cpu().numpy()
            pred_batch = output.pred.detach().cpu().numpy()

            assert ipt_batch.shape[0] == tgt_batch.shape[0], (ipt_batch.shape, tgt_batch.shape)
            assert len(tgt_batch.shape) == 5, tgt_batch.shape
            assert tgt_batch.shape[1] == 1, tgt_batch.shape

            has_aux = hasattr(output, "aux_tgt")
            if has_aux:
                aux_tgt_batch = numpy.stack([yy.cpu().numpy() for yy in output.aux_tgt])
                aux_pred_batch = numpy.stack([yy.detach().cpu().numpy() for yy in output.aux_pred])
                assert ipt_batch.shape[0] == aux_tgt_batch.shape[0], (ipt_batch.shape, aux_tgt_batch.shape)
                assert len(aux_tgt_batch.shape) == 5, aux_tgt_batch.shape
                assert aux_tgt_batch.shape[1] == 1, aux_tgt_batch.shape
            else:
                aux_tgt_batch = None
                aux_pred_batch = None

            has_voxel_losses = output.voxel_losses is not None and not all(vl is None for vl in output.voxel_losses)
            ncols = 5 + 2 * int(has_aux)
            if has_voxel_losses:
                ncols += len(output.voxel_losses)

            nrows = ipt_batch.shape[0]
            fig, ax = plt.subplots(ncols=ncols, nrows=nrows, squeeze=False, figsize=(ncols * 3, nrows * 3))

            def make_subplot(
                ax_list, title: str, img, boxes: Iterable[Box] = tuple(), side_view=None, with_colorbar=True
            ):
                global col
                ax = ax_list[col]
                if title:
                    ax.set_title(title)

                if side_view is not None:
                    img = numpy.concatenate(
                        [
                            img,
                            numpy.full(shape=(img.shape[0], 1), fill_value=side_view.max()),
                            numpy.repeat(side_view, 3, axis=1),
                        ],
                        axis=1,
                    )

                im = ax.imshow(img, cmap=ListedColormap(turbo_colormap_data))
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.axis("off")
                for box in boxes:
                    box.apply_to_ax(ax)

                if with_colorbar:
                    # from https://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
                    # create an axes on the right side of ax. The width of cax will be 5%
                    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="3%", pad=0.03)
                    fig.colorbar(im, cax=cax)
                    # ax.set_title(f"min-{img.min():.2f}-max-{img.max():.2f}")  # taking too much space!

                col += 1

            global col
            for i, (ib, tb, pb) in enumerate(zip(ipt_batch, tgt_batch, pred_batch)):
                col = 0
                make_subplot(ax[i], "", ib[0])
                make_subplot(ax[i], "target", tb[0].max(axis=0), boxes=boxes, side_view=tb[0].max(axis=2).T)
                make_subplot(ax[i], "prediction", pb[0].max(axis=0), boxes=boxes, side_view=pb[0].max(axis=2).T)
                rel_diff = (numpy.abs(pb - tb) / tb + 1e-6)[0]
                abs_diff = numpy.abs(pb - tb)[0]
                make_subplot(ax[i], "rel diff", rel_diff.max(axis=0), side_view=rel_diff.max(axis=2).T)
                make_subplot(ax[i], "abs_diff", abs_diff.max(axis=0), side_view=abs_diff.max(axis=2).T)

            if has_aux:
                col_so_far = col
                for i, (atb, apb) in enumerate(zip(aux_tgt_batch, aux_pred_batch)):
                    col = col_so_far
                    make_subplot(ax[i], "aux tgt", atb[0].max(axis=0), boxes=boxes)
                    make_subplot(ax[i], "aux pred", apb[0].max(axis=0), boxes=boxes)

            if has_voxel_losses:
                col_so_far = col
                voxel_losses = numpy.stack([ll.detach().cpu().numpy() for ll in output.voxel_losses])
                for loss_nr, vl_batch in enumerate(voxel_losses):
                    for i, vl in enumerate(vl_batch):
                        col = col_so_far
                        make_subplot(
                            ax[i],
                            f"voxel loss {loss_nr}",
                            vl.max(axis=0).max(axis=0),
                            boxes=boxes,
                            side_view=vl.max(axis=0).max(axis=2).T,
                        )

            fig.subplots_adjust(hspace=0, wspace=0, bottom=0, top=1, left=0, right=1)
            fig.tight_layout()

            writer.add_figure(f"{engine.name}/in_out", fig, step)

        def log_eval(engine: TunedEngine):
            metrics = engine.state.metrics
            self.logger.info("%s - Epoch: %d  Avg loss: %.3f", engine.name, engine.state.epoch, metrics[LOSS_NAME])

            # available_metrics = [LOSS_NAME, MSSSIM_NAME, SSIM_NAME, PSNR_NAME, NRMSE_NAME]
            # if len(engine.state.loss) > 1:
            #     for i in range(len(engine.state.loss)):
            #         available_metrics.append(f"{LOSS_NAME}-{i}")
            #
            # if engine.state.aux_loss is not None:
            #     available_metrics.append(AUX_LOSS_NAME)
            #     for i in range(len(engine.state.aux_loss)):
            #         available_metrics.append(f"{AUX_LOSS_NAME}-{i}")
            #
            # if BEAD_PRECISION_RECALL in met:
            #     bprecision, brecall = met[BEAD_PRECISION_RECALL]
            #     if not numpy.isnan(bprecision):
            #         met[BEAD_PRECISION] = bprecision
            #         available_metrics.append(BEAD_PRECISION)
            #
            #     if not numpy.isnan(brecall):
            #         met[BEAD_RECALL] = brecall
            #         available_metrics.append(BEAD_RECALL)

            def log_metric(m: str):
                writer.add_scalar(f"{engine.name}/{m}", metrics[m], trainer.state.epoch)
                metric_log_file = self.config.log.dir / engine.name / f"{m.lower()}.txt"
                with metric_log_file.open(mode="a") as file:
                    file.write(f"{trainer.state.epoch}\t{metrics[m]}\n")

            for m in metrics:
                print('here:', m)

            print()
            [log_metric(m) for m in metrics]
            log_images(engine, trainer.state.epoch)

            # for i, yy in enumerate(experiment.state.output["y"]):
            #     yy = yy.cpu().numpy()
            #     assert len(yy.shape) == 4, yy.shape
            #     assert yy.shape[0] == 1, yy.shape
            #     writer.add_images(f"{name}/output{i}", yy.max(axis=1), step, dataformats="CHW")

        train_evaluator.add_event_handler(Events.COMPLETED, log_eval)
        validator.add_event_handler(Events.COMPLETED, log_eval)
        tester.add_event_handler(Events.COMPLETED, log_eval)

        def add_save_result(engine: EvalEngine):
            @engine.on(Events.STARTED)
            def setup_save_result(engine: EvalEngine):
                engine.state.sequential_sample_nr_for_result = 0

                concat_kwargs = {}
                if isinstance(engine.state.dataloader.dataset, ConcatDataset):
                    cumsum = engine.state.dataloader.dataset.cumulative_sizes
                    concat_kwargs["cumsum"] = cumsum
                    if engine.state.dataloader == engine.data_config.data_loader:
                        concat_kwargs["subfolders"] = [e.name for e in engine.data_config.entries]
                    else:
                        concat_kwargs["subfolders"] = [str(cs) for cs in cumsum]

                engine.state.result = Result(
                    self.config.log.dir / engine.name / "input",
                    self.config.log.dir / engine.name / "target",
                    self.config.log.dir / engine.name / "prediction",
                    **concat_kwargs,
                )

            @engine.on(Events.ITERATION_COMPLETED)
            def save_result(engine: EvalEngine):
                output: Output = engine.state.output
                start = engine.state.sequential_sample_nr_for_result
                engine.state.result.update(
                    *[batch.detach().cpu().numpy() for batch in [output.ipt, output.tgt, output.pred]], at=start
                )
                engine.state.sequential_sample_nr_for_result += output.ipt.shape[0]

        add_save_result(tester)

        @trainer.on(Events.ITERATION_COMPLETED)
        def log_training_iteration(engine: TrainEngine):
            iteration = engine.state.iteration
            it_in_epoch = (iteration - 1) % len(engine.state.dataloader) + 1
            if (
                engine.config.log.log_scalars_every[1] == Events.ITERATION_COMPLETED.value
                and it_in_epoch % engine.config.log.log_scalars_every[0] == 0
            ):
                log_train_scalars(engine, iteration)

            if (
                engine.config.log.log_images_every[1] == Events.ITERATION_COMPLETED.value
                and it_in_epoch % engine.config.log.log_images_every[0] == 0
            ):
                log_images(engine, iteration)

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_training_epoch(engine: TrainEngine):
            epoch = engine.state.epoch
            if (
                engine.config.log.log_scalars_every[1] == Events.EPOCH_COMPLETED.value
                and epoch % engine.config.log.log_scalars_every[0] == 0
            ):
                log_train_scalars(engine, epoch)

            if (
                engine.config.log.log_images_every[1] == Events.EPOCH_COMPLETED.value
                and epoch % engine.config.log.log_images_every[0] == 0
            ):
                log_images(engine, epoch)

        @trainer.on(Events.EPOCH_COMPLETED)
        def validate(engine: TrainEngine):
            if engine.state.epoch % engine.config.train.validate_every_nth_epoch == 0:
                train_evaluator.run()
                validator.run()

        @trainer.on(Events.COMPLETED)
        def log_test_results(engine: TrainEngine):
            if saver._saved:
                score, file_list = saver._saved[-1]
                engine.model.load_state_dict(
                    torch.load(file_list[0], map_location=next(engine.model.parameters()).device)
                )

            tester.run()

        trainer.run(max_epochs=self.max_num_epochs)
        writer.close()
        if self.config_path is not None:
            with self.config_path.with_suffix(".ran_on.txt").open("a+") as f:
                if self.config.log.commit_hash not in f.read().split("\n"):
                    f.write(self.config.log.commit_hash)
