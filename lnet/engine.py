from logging import Logger
from typing import List, Callable, Tuple, Optional, Any, TypeVar

import torch

from ignite.engine import Engine, Events
from ignite.handlers import TerminateOnNan
from torch.utils.data import DataLoader, ConcatDataset, SubsetRandomSampler

from lnet.config import Config, DataConfig
from lnet.datasets import SubsetSequentialSampler
from lnet.output import Output

from lnet.utils.transforms import lightfield_from_channel, EdgeCrop
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
from lnet.utils.metrics import NRMSE, PSNR, SSIM, MSSSIM
from lnet.utils.metrics.beads import BeadPrecisionRecall


TunedEngineType = TypeVar("TunedEngineType", bound="TunedEngine")


class TunedEngine(Engine):
    data_loader: DataLoader = None

    def __init__(
        self,
        process_function: Callable[[TunedEngineType, Any], Output],
        config: Config,
        logger: Logger,
        model: torch.nn.Module,
        data_config: DataConfig,
    ):
        super().__init__(process_function)
        self.config = config
        self.logger = logger
        self.model = model
        self.run_count = 0

        (config.log.dir / data_config.name).mkdir(parents=False, exist_ok=False)

        @torch.no_grad()
        def get_yx_yy(x_shape: Tuple[int, int]) -> Tuple[int, int]:
            xx, xy = x_shape
            xc = config.model.nnum ** 2
            xx = xx // config.model.nnum
            xy = xy // config.model.nnum
            dummy_pred = model(
                torch.randn((1, xc, xx, xy), dtype=getattr(torch, config.model.precision), device=next(model.parameters()).device)
            )
            if isinstance(dummy_pred, tuple):
                if len(dummy_pred) == 2:
                    dummy_pred, dummy_pred_aux = dummy_pred
                    assert dummy_pred.shape == dummy_pred_aux.shape
                else:
                    raise NotImplementedError

            n_pred, c_pred, zout_pred, yx, yy = dummy_pred.shape
            assert n_pred == 1
            assert c_pred == 1

            if hasattr(model, "get_target_crop"):
                crop = model.get_target_crop()
                if crop is not None:
                    cx, cy = crop
                    yx += 2 * cx
                    yy += 2 * cy

            return yx, yy

        self.dataset, z_out, ipaths = data_config.factory.create_dataset(
            get_yx_yy=get_yx_yy, transforms=data_config.transforms
        )

        transforms = list(data_config.transforms)
        if hasattr(model, "get_target_crop"):
            transforms.append(EdgeCrop(model.get_target_crop(), apply_to=[1]))

        def get_full_data_indices(concat_dataset: ConcatDataset, data_indices: Optional[List[List[int]]]) -> List[int]:
            if data_indices is None:
                return list(range(len(concat_dataset)))
            else:
                full_data_indices = []
                before = 0
                for ds, tdi in zip(concat_dataset.datasets, data_indices):
                    if tdi is None:
                        full_data_indices += [before + i for i in range(len(ds))]
                    elif tdi:
                        assert max(tdi) < len(ds), (max(tdi), len(ds))
                        full_data_indices += [before + i for i in tdi]

                    before += len(ds)

                return full_data_indices

        self.full_data_indices = get_full_data_indices(self.dataset, data_config.indices)

        @self.on(Events.STARTED)
        def prepare_engine(engine: TunedEngine):
            engine.run_count += 1
            engine.state.loss_fn = engine.config.train.loss_fn(engine, **engine.config.train.loss_fn_kwargs)
            if engine.config.train.loss_fn_aux is None:
                engine.state.loss_fn_aux = None
            else:
                engine.state.loss_fn_aux = engine.config.train.loss_fn_aux(engine, **engine.config.train.loss_fn_aux_kwargs)

        @self.on(Events.COMPLETED)
        def log_compute_time(engine: TunedEngine):
            mins, secs = divmod(engine.state.compute_time / max(1, engine.state.iteration), 60)
            msecs = (secs % 1) * 1000
            hours, mins = divmod(mins, 60)
            engine.logger.info(
                "%s run on %d mini-batches completed in %.2f with avg compute time %02d:%02d:%02d:%03d",
                engine.state.name,
                len(engine.state.dataloader),
                engine.state.compute_time,
                hours,
                mins,
                secs,
                msecs,
            )

    def run(self, data: Optional[DataLoader] = None, max_epochs: int = 1):
        assert not (data and self.data_loader)
        super().run(data=data or self.data_loader, max_epochs=max_epochs)


class TrainEngine(TunedEngine):
    def __init__(
        self,
        process_function: Callable[["TrainEngine", Any], Output],
        config: Config,
        logger: Logger,
        model: torch.nn.Module,
    ):
        super().__init__(
            process_function=process_function, config=config, logger=logger, model=model, data_config=config.train_data
        )

        self.add_event_handler(
            Events.ITERATION_COMPLETED, TerminateOnNan(output_transform=lambda output: output.__dict__)
        )

        @self.on(Events.STARTED)
        def prepare_engine(engine: "TrainEngine"):
            engine.state.optimizer = config.train.optimizer(model.parameters())
            engine.state.focus_weight = config.train.focus.initial_weight

        self.data_loader = DataLoader(
            self.dataset,
            batch_size=config.train.batch_size,
            pin_memory=True,
            num_workers=5,
            sampler=SubsetRandomSampler(self.full_data_indices),
        )


class EvalEngine(TunedEngine):
    def __init__(
        self,
        process_function: Callable[["EvalEngine", Any], Output],
        config: Config,
        logger: Logger,
        model: torch.nn.Module,
        data_config: DataConfig,
    ):
        super().__init__(
            process_function=process_function, config=config, logger=logger, model=model, data_config=data_config
        )

        MSSSIM().attach(self, MSSSIM_NAME)
        NRMSE().attach(self, NRMSE_NAME)
        PSNR(data_range=2.5).attach(self, PSNR_NAME)
        SSIM().attach(self, SSIM_NAME)
        if config.log.log_bead_precision_recall:
            BeadPrecisionRecall(dist_threshold=config.log.log_bead_precision_recall_threshold).attach(
                self, BEAD_PRECISION_RECALL
            )

        self.data_loader = DataLoader(
            self.dataset,
            batch_size=config.eval.batch_size,
            pin_memory=True,
            num_workers=5,
            sampler=SubsetSequentialSampler(self.full_data_indices),
        )
