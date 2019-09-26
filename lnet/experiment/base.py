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
from inferno.io.transform import Transform
from matplotlib import patches
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pathlib import Path

from scipy.special._ufuncs import expit
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, SequentialSampler, SubsetRandomSampler
from typing import Union, Optional, List, Dict, Callable, Type, Any, Tuple, Sequence, Iterable

from lnet.experiment.config import Config
from lnet.utils.data_transform import lightfield_from_channel, EdgeCrop
from lnet.utils.datasets import DatasetFactory, SubsetSequentialSampler, Result
from lnet.utils.metrics import NRMSE, PSNR, SSIM, MSSSIM
from lnet.utils.metrics.beads import BeadPrecisionRecall
from lnet.utils.plotting import turbo_colormap_data

eps_for_precision = {torch.half: 1e-4, torch.float: 1e-8}
torch_dtype_to_inferno = {torch.float: "float", torch.float32: "float", torch.half: "half", torch.float16: "half"}


LOSS_NAME = "Loss"
AUX_LOSS_NAME = "AuxLoss"
NRMSE_NAME = "NRMSE"
PSNR_NAME = "PSNR"
SSIM_NAME = "SSIM"
MSSSIM_NAME = "MS-SSIM"
BEAD_PRECISION_RECALL = "Bead-Precision-Recall"
BEAD_PRECISION = "Bead-Precision"
BEAD_RECALL = "Bead-Recall"


class ExperimentBase:
    Model: Union[Type[torch.nn.Module], Callable[..., torch.nn.Module]]
    additional_model_kwargs: Dict[str, Any]
    nnum: int

    train_dataset_factory: DatasetFactory
    valid_dataset_factory: Optional[DatasetFactory] = None
    test_dataset_factory: Optional[DatasetFactory] = None

    train_data_range: Optional[Union[range, Iterable[int]]] = None
    train_eval_data_range: Optional[Union[range, Iterable[int]]] = None
    valid_data_range: Optional[Union[range, Iterable[int]]] = None
    test_data_range: Optional[Union[range, Iterable[int]]] = None

    batch_size: int
    eval_batch_size: int

    train_transforms: List[Union[str, Transform]]
    valid_transforms: List[Union[str, Transform]]
    test_transforms: List[Union[str, Transform]]

    optimizer_cls: Callable
    precision: torch.dtype
    optimizer_kwargs: Dict[str, Any]
    max_num_epochs: int
    score_function: Callable[[Engine], float]

    checkpoint: Optional[Path]
    config: Config
    add_in_name: Optional[str] = None

    def __init__(
        self,
        config: Config,
        loss_fn: Union[torch.nn.Module, List[Tuple[float, torch.nn.Module]]],
        aux_loss_fn: Optional[Union[torch.nn.Module, List[Tuple[float, torch.nn.Module]]]] = None,
        checkpoint: Optional[Path] = None,
        dist_threshold: float = 5.0,
        pre_loss_transform: Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]] = lambda *x: x,
        pre_aux_loss_transform: Callable[
            [torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]
        ] = lambda *x: x,
    ):
        self.config = config
        # # make sure everything is commited
        # git_status_cmd = pbs3.git.status("--porcelain")
        # git_status = git_status_cmd.stdout
        # assert not git_status, git_status  # uncommited changes exist
        self.dist_threshold = dist_threshold
        self.pre_loss_transform = pre_loss_transform
        self.pre_aux_loss_transform = pre_aux_loss_transform

        if not isinstance(loss_fn, list):
            loss_fn = [(1.0, loss_fn)]

        self.loss_fn: List[torch.nn.Module] = loss_fn

        if aux_loss_fn is not None and not isinstance(aux_loss_fn, list):
            aux_loss_fn = [(1.0, aux_loss_fn)]

        self.aux_loss_fn: Optional[List[torch.nn.Module]] = aux_loss_fn

        self.checkpoint = checkpoint

        self.logger = logging.getLogger(config.log.time_stamp)

    def get_yx_yy(self, x_shape=Tuple[int, int]) -> Tuple[int, int]:
        xx, xy = x_shape
        xc = self.nnum ** 2
        xx = xx // self.nnum
        xy = xy // self.nnum
        with torch.no_grad():
            if torch.cuda.is_available():
                self.model.cuda()
                device = "cuda"
            else:
                device = "cpu"

            dummy_pred = self.model(torch.randn((1, xc, xx, xy), dtype=self.precision, device=device))
            if isinstance(dummy_pred, tuple):
                if len(dummy_pred) == 2:
                    dummy_pred, dummy_pred_aux = dummy_pred
                    assert dummy_pred.shape == dummy_pred_aux.shape
                else:
                    raise NotImplementedError

            n_pred, c_pred, zout_pred, yx, yy = dummy_pred.shape
            assert n_pred == 1
            assert c_pred == 1
            assert zout_pred == self.z_out, (zout_pred, self.z_out)

        if hasattr(self.model, "get_target_crop"):
            crop = self.model.get_target_crop()
            if crop is not None:
                cx, cy = crop
                yx += 2 * cx
                yy += 2 * cy

        return yx, yy

    def test(self):
        self.max_num_epochs = 0
        self.run()

    def run(self):
        self.z_out = self.train_dataset_factory.get_z_out()

        devices = list(range(torch.cuda.device_count()))
        if devices:
            device = torch.device("cuda", devices[0])
        else:
            device = torch.device("cpu")

        self.model = self.Model(nnum=self.nnum, z_out=self.z_out, **self.additional_model_kwargs).to(
            device=device, dtype=self.precision
        )
        # todo: warmstart from checkpoints / load from checkpoints
        if self.checkpoint is not None:
            state = torch.load(self.checkpoint, map_location=device)
            self.model.load_state_dict(state, strict=False)

        optimizer = self.optimizer_cls(self.model.parameters(), **self.optimizer_kwargs)

        if hasattr(self.model, "get_target_crop"):
            self.train_transforms.append(EdgeCrop(self.model.get_target_crop(), apply_to=[1]))
            self.valid_transforms.append(EdgeCrop(self.model.get_target_crop(), apply_to=[1]))
            self.test_transforms.append(EdgeCrop(self.model.get_target_crop(), apply_to=[1]))

        train_dataset, z_out_train, _ = self.train_dataset_factory.create_dataset(
            get_yx_yy=self.get_yx_yy, transforms=self.train_transforms
        )
        train_eval_dataset, _, train_ipaths = self.train_dataset_factory.create_dataset(
            get_yx_yy=self.get_yx_yy, transforms=self.valid_transforms
        )
        assert self.z_out == z_out_train, (self.z_out, z_out_train)
        valid_dataset, z_out_valid, valid_ipaths = (
            (None, self.z_out, [[]])
            if self.valid_dataset_factory is None
            else self.valid_dataset_factory.create_dataset(get_yx_yy=self.get_yx_yy, transforms=self.valid_transforms)
        )
        assert self.z_out == z_out_valid, (self.z_out, z_out_valid)

        test_dataset, z_out_test, test_ipaths = (
            (None, self.z_out, [[]])
            if self.test_dataset_factory is None
            else self.test_dataset_factory.create_dataset(get_yx_yy=self.get_yx_yy, transforms=self.test_transforms)
        )
        assert self.z_out == z_out_test, (self.z_out, z_out_test)

        if self.train_data_range is None:
            self.train_data_range = range(len(train_dataset))

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=5,
            sampler=SubsetRandomSampler(list(self.train_data_range)),
        )

        if self.train_eval_data_range is None:
            self.train_eval_data_range = range(len(train_eval_dataset))

        train_ipaths = train_ipaths[self.train_eval_data_range.start : self.train_eval_data_range.stop]
        train_loader_eval = DataLoader(
            train_eval_dataset,
            batch_size=self.eval_batch_size,
            pin_memory=True,
            num_workers=3,
            # despite 'shuffle=False' the order is not always the same, trying out SequentialSampler...
            sampler=SubsetSequentialSampler(self.train_eval_data_range),
        )

        if self.valid_data_range is None and valid_dataset is not None:
            self.valid_data_range = range(len(valid_dataset))

        valid_ipaths = valid_ipaths[self.valid_data_range.start : self.valid_data_range.stop]
        valid_loader = (
            None
            if valid_dataset is None
            else DataLoader(
                valid_dataset,
                batch_size=self.eval_batch_size,
                pin_memory=True,
                num_workers=3,
                sampler=SubsetSequentialSampler(self.valid_data_range),
            )
        )

        if self.test_data_range is None and test_dataset is not None:
            self.test_data_range = range(len(test_dataset))

        test_ipaths = test_ipaths[self.test_data_range.start : self.test_data_range.stop]
        test_loader = (
            None
            if test_dataset is None
            else DataLoader(
                test_dataset,
                batch_size=self.eval_batch_size,
                pin_memory=True,
                num_workers=5,
                sampler=SubsetSequentialSampler(self.test_data_range),
            )
        )

        # tensorboardX
        writer = SummaryWriter(self.config.log.dir.as_posix())
        # data_loader_iter = iter(train_loader)
        # x, y = next(data_loader_iter)
        # try:
        #     writer.add_graph(self.model, x.to(torch.device("cuda")))
        # except Exception as e:
        #     self.logger.warning("Failed to save model graph...")
        #     self.logger.exception(e)

        # ignite
        class CustomEvents(Enum):
            VALIDATION_DONE = "validation_done_event"

        @dataclass
        class Output:
            ipt: torch.Tensor
            tgt: torch.Tensor
            aux_tgt: torch.Tensor
            pred: torch.Tensor
            aux_pred: torch.Tensor
            loss: torch.Tensor
            aux_loss: torch.Tensor
            losses: List[torch.Tensor]
            aux_losses: List[torch.Tensor]

        result_dir = self.config.log.dir / "result"

        class TunedEngine(Engine):
            def __init__(self, process_function):
                super().__init__(process_function)
                self.named_run_counts = {}
                self.add_event_handler(Events.STARTED, self.prepare_engine)
                self.add_event_handler(Events.COMPLETED, self.log_compute_time)

            @property
            def run_count(self):
                return self.named_run_counts.get(self.state.name, 1)

            @staticmethod
            def prepare_engine(engine: "TunedEngine"):
                engine.state.compute_time = 0
                eval = True
                if engine.state.dataloader == train_loader:
                    engine.state.name = "in_training"
                    eval = False
                elif engine.state.dataloader == train_loader_eval:
                    engine.state.name = "training"
                    engine.state.ipaths = train_ipaths
                elif engine.state.dataloader == valid_loader:
                    engine.state.name = "validation"
                    engine.state.ipaths = valid_ipaths
                elif engine.state.dataloader == test_loader:
                    engine.state.name = "test"
                    engine.state.ipaths = test_ipaths
                else:
                    raise NotImplementedError

                if eval:
                    (result_dir / engine.state.name).mkdir(parents=True, exist_ok=True)
                    if engine.state.ipaths:
                        engine.state.ipaths_log = numpy.full(
                            (len(engine.state.ipaths), len(engine.state.ipaths[0])), -1, dtype=numpy.float32
                        )
                        engine.state.target_ipaths_log = numpy.full(
                            (len(engine.state.ipaths), len(engine.state.ipaths[0])), -1, dtype=numpy.float32
                        )
                    else:
                        engine.state.ipaths_log = numpy.empty((0, 0))
                        engine.state.target_ipaths_log = numpy.empty((0, 0))

                engine.named_run_counts[engine.state.name] = engine.run_count + 1

            @staticmethod
            def log_compute_time(engine: "TunedEngine"):
                mins, secs = divmod(engine.state.compute_time / max(1, engine.state.iteration), 60)
                msecs = (secs % 1) * 1000
                hours, mins = divmod(mins, 60)
                self.logger.info(
                    "%s run on %d mini-batches completed in %.2f with avg compute time %02d:%02d:%02d:%03d",
                    engine.state.name,
                    len(engine.state.dataloader),
                    engine.state.compute_time,
                    hours,
                    mins,
                    secs,
                    msecs,
                )

        def training_step(engine, batch) -> Output:
            self.model.train()
            start = time.time()
            optimizer.zero_grad()
            has_aux = len(batch) == 3
            if has_aux:
                ipt, tgt, aux_tgt = batch
                aux_tgt = convert_tensor(aux_tgt, device=device, non_blocking=False)
            else:
                ipt, tgt = batch
                aux_tgt = None

            ipt = convert_tensor(ipt, device=device, non_blocking=False)
            tgt = convert_tensor(tgt, device=device, non_blocking=False)
            pred = self.model(ipt)
            if has_aux:
                pred, aux_pred = pred
                aux_losses = [w * lf(*self.pre_aux_loss_transform(aux_pred, tgt)) for w, lf in self.aux_loss_fn]
                aux_loss = sum(aux_losses)
            else:
                aux_pred = None
                aux_losses = None
                aux_loss = None

            losses = [w * lf(*self.pre_loss_transform(pred, tgt)) for w, lf in self.loss_fn]
            total_loss = sum(losses)
            loss = total_loss
            if has_aux:
                total_loss += aux_loss

            total_loss.backward()
            optimizer.step()
            engine.state.compute_time += time.time() - start
            return Output(
                ipt=ipt,
                tgt=tgt,
                aux_tgt=aux_tgt,
                pred=pred,
                aux_pred=aux_pred,
                loss=loss,
                aux_loss=aux_loss,
                losses=losses,
                aux_losses=aux_losses,
            )

        trainer = TunedEngine(training_step)
        trainer.add_event_handler(
            Events.ITERATION_COMPLETED, TerminateOnNan(output_transform=lambda output: output.__dict__)
        )

        def inference_step(engine, batch) -> Output:
            self.model.eval()
            with torch.no_grad():
                start = time.time()
                if len(batch) == 3:
                    ipt, tgt, aux_tgt = batch
                    aux_tgt = convert_tensor(aux_tgt, device=device, non_blocking=False)
                else:
                    ipt, tgt = batch
                    aux_tgt = None

                ipt = convert_tensor(ipt, device=device, non_blocking=False)
                tgt = convert_tensor(tgt, device=device, non_blocking=False)
                pred = self.model(ipt)
                engine.state.compute_time += time.time() - start

                if isinstance(pred, tuple):
                    pred, aux_pred = pred
                    aux_losses = [w * lf(aux_pred, aux_tgt) for w, lf in self.aux_loss_fn]
                    aux_loss = sum(aux_losses)
                else:
                    aux_pred = None
                    aux_losses = None
                    aux_loss = None

                losses = [w * lf(pred, tgt) for w, lf in self.loss_fn]
                loss = sum(losses)
                return Output(
                    ipt=ipt,
                    tgt=tgt,
                    aux_tgt=aux_tgt,
                    pred=pred,
                    aux_pred=aux_pred,
                    loss=loss,
                    aux_loss=aux_loss,
                    losses=losses,
                    aux_losses=aux_losses,
                )

        evaluator = TunedEngine(inference_step)
        evaluator.register_events(*CustomEvents)

        saver = ModelCheckpoint(
            (self.config.log.dir / "models").as_posix(),
            "v0",
            score_function=self.score_function,
            n_saved=1,
            create_dir=True,
            save_as_state_dict=True,
        )
        evaluator.add_event_handler(
            CustomEvents.VALIDATION_DONE, saver, {"model": self.model}
        )  # , "optimizer": optimizer})
        stopper = EarlyStopping(patience=200, score_function=self.score_function, trainer=trainer)
        evaluator.add_event_handler(CustomEvents.VALIDATION_DONE, stopper)

        Loss(loss_fn=lambda loss, _: loss, output_transform=lambda out: (out.loss, out.ipt)).attach(
            evaluator, LOSS_NAME
        )
        if len(self.loss_fn) > 1:
            for i in range(len(self.loss_fn)):
                Loss(loss_fn=lambda loss, _: loss, output_transform=lambda out, j=i: (out.losses[j], out.ipt)).attach(
                    evaluator, f"{LOSS_NAME}-{i}"
                )

        if self.aux_loss_fn is not None:
            Loss(loss_fn=lambda loss, _: loss, output_transform=lambda out: (out.aux_loss, out.ipt)).attach(
                evaluator, AUX_LOSS_NAME
            )
            for i in range(len(self.aux_loss_fn)):
                Loss(
                    loss_fn=lambda loss, _: loss, output_transform=lambda out, j=i: (out.aux_losses[j], out.ipt)
                ).attach(evaluator, f"{AUX_LOSS_NAME}-{i}")

        MSSSIM().attach(evaluator, MSSSIM_NAME)
        NRMSE().attach(evaluator, NRMSE_NAME)
        PSNR(data_range=2.5).attach(evaluator, PSNR_NAME)
        SSIM().attach(evaluator, SSIM_NAME)
        if self.config.log.log_bead_precision_recall:
            BeadPrecisionRecall(dist_threshold=self.dist_threshold).attach(evaluator, BEAD_PRECISION_RECALL)

        def log_train_scalars(engine: Engine, step: int):
            writer.add_scalar(f"{engine.state.name}/loss", engine.state.output.loss, step)

        class Box:
            def __init__(self, slice_x: slice, slice_y: slice, color: str):
                self.slice_x = slice_x
                self.slice_y = slice_y
                self.color = color

            def apply_to_ax(self, ax):
                box = patches.Rectangle(
                    (self.slice_x.start, self.slice_y.start),
                    self.slice_x.stop - self.slice_x.start,
                    self.slice_y.stop - self.slice_y.start,
                    linewidth=1,
                    edgecolor=self.color,
                    facecolor="none",
                )
                ax.add_patch(box)

        def log_images(engine: Engine, step: int, boxes: Iterable[Box] = tuple()):
            output: Output = engine.state.output
            ipt_batch = numpy.stack([lightfield_from_channel(xx, nnum=self.nnum) for xx in output.ipt.cpu().numpy()])

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

        @evaluator.on(Events.COMPLETED)
        def log_eval(engine: TunedEngine):
            met = evaluator.state.metrics
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
                metric_log_file = result_dir / engine.state.name / f"{metric.lower()}.txt"
                with metric_log_file.open(mode="a") as file:
                    file.write(f"{trainer.state.epoch}\t{met[metric]}\n")

            [log_metric(metric) for metric in available_metrics]

            boxes = []
            if engine.state.ipaths_log.shape[1]:
                fig, ax = plt.subplots()
                n_times = engine.state.ipaths_log.shape[0]
                # ax.set_xlim([0, n_times - 1])
                ax.set_ylim([0, 1.0])

                class ColorSelection:
                    def __init__(self, colors: Sequence[str]):
                        self.colors = colors

                    def __getitem__(self, item: int):
                        return self.colors[item % len(self.colors)]

                colors = ColorSelection(["b", "g", "r", "c", "m", "y"])

                # log path data to file as well
                this_path_dir = result_dir / engine.state.name / "paths" / f"epoch{trainer.state.epoch}"
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

        test_result = Result(result_dir / "test" / "prediction", result_dir / "test" / "target")

        @evaluator.on(Events.ITERATION_COMPLETED)
        def eval_iteration(engine: Engine):
            output: Output = engine.state.output
            start = ((engine.state.iteration - 1) % len(engine.state.dataloader)) * self.eval_batch_size
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
            it_in_epoch = (iteration - 1) % len(train_loader) + 1
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
            if engine.state.epoch % self.config.log.validate_every_nth_epoch == 0:
                # evaluate on training data
                evaluator.run(train_loader_eval)

                # evaluate on validation data
                if valid_loader is not None:
                    evaluator.run(valid_loader)

                evaluator.fire_event(CustomEvents.VALIDATION_DONE)

        @trainer.on(Events.COMPLETED)
        def log_test_results(engine: TunedEngine):
            if test_loader is not None:
                if saver._saved:
                    score, file_list = saver._saved[-1]
                    self.model.load_state_dict(
                        torch.load(file_list[0], map_location=next(self.model.parameters()).device)
                    )

                evaluator.run(test_loader)

        trainer.run(train_loader, max_epochs=self.max_num_epochs)
        writer.close()

    def export_test_predictions(self, output_path: Optional[str] = None):
        raise NotImplementedError
