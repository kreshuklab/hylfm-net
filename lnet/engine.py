import warnings
from logging import Logger
from typing import Callable, Optional, Any, TypeVar

import torch

from ignite.engine import Engine, Events
from ignite.handlers import TerminateOnNan
from torch.utils.data import DataLoader

from lnet.config.base import Config
from lnet.config.data import DataConfig
from lnet.output import Output
from lnet.metrics.output import OutputMetric

from lnet.metrics import LOSS_NAME, AUX_LOSS_NAME, NRMSE_NAME, PSNR_NAME, SSIM_NAME, MSSSIM_NAME, BEAD_PRECISION_RECALL
from lnet.metrics import NRMSE, PSNR, SSIM, MSSSIM
from lnet.metrics.beads import BeadPrecisionRecall


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
        self.data_config = data_config

        self.name = data_config.category.value
        self.run_count = 0

        (config.log.dir / self.name).mkdir(parents=False, exist_ok=False)

        self.add_event_handler(Events.STARTED, self.prepare_engine)
        self.add_event_handler(Events.COMPLETED, self.log_compute_time)

    @staticmethod
    def prepare_engine(engine: TunedEngineType):
        engine.state.compute_time = 0.0
        engine.run_count += 1
        if engine.config.train is None:
            engine.state.loss = []
            engine.state.aux_loss = None
        else:
            engine.state.loss = engine.config.train.loss(engine, engine.config.train.loss_kwargs)
            engine.state.aux_loss = engine.config.train.aux_loss(engine, engine.config.train.aux_loss_kwargs)

        train = isinstance(engine, TrainEngine)
        for w, l in engine.state.loss:
            l.train(mode=train)

        if engine.state.aux_loss is not None:
            for w, l in engine.state.aux_loss:
                l.train(mode=train)

    @staticmethod
    def log_compute_time(engine: TunedEngineType):
        mins, secs = divmod(engine.state.compute_time / max(1, engine.state.iteration), 60)
        msecs = (secs % 1) * 1000
        hours, mins = divmod(mins, 60)
        engine.logger.info(
            "%s run on %d mini-batches completed in %.2f s with avg compute time %02d:%02d:%02d:%03d",
            engine.name,
            len(engine.state.dataloader),
            engine.state.compute_time,
            hours,
            mins,
            secs,
            msecs,
        )

    def run(self, data: Optional[DataLoader] = None, max_epochs: int = 1):
        data = data or self.data_config.data_loader
        if data:
            super().run(data=data, max_epochs=max_epochs)
        else:
            warnings.warn(f"no data provided for {self.name}")


class TrainEngine(TunedEngine):
    def __init__(
        self,
        process_function: Callable[["TrainEngine", Any], Output],
        config: Config,
        logger: Logger,
        model: torch.nn.Module,
    ):
        super().__init__(
            process_function=process_function, config=config, logger=logger, model=model, data_config=config.train.data
        )

        self.add_event_handler(
            Events.ITERATION_COMPLETED, TerminateOnNan(output_transform=lambda output: output.__dict__)
        )

    @staticmethod
    def prepare_engine(engine: TunedEngineType):
        super().prepare_engine(engine)
        engine.state.optimizer = engine.config.train.optimizer(engine.model.parameters())


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
        if data_config.z_out is not None and not any(e.info.y_path is None for e in data_config.entries):
            MSSSIM().attach(self, MSSSIM_NAME)
            NRMSE().attach(self, NRMSE_NAME)
            PSNR(data_range=2.5).attach(self, PSNR_NAME)
            SSIM().attach(self, SSIM_NAME)
            if config.log.log_bead_precision_recall:
                BeadPrecisionRecall(dist_threshold=config.log.log_bead_precision_recall_threshold).attach(
                    self, BEAD_PRECISION_RECALL
                )

    @staticmethod
    def prepare_engine(engine: TunedEngineType):
        super().prepare_engine(engine)
        OutputMetric(out_to_metric=lambda out: out.loss).attach(engine, LOSS_NAME)
        if len(engine.state.loss) > 1:
            for i in range(len(engine.state.loss)):
                OutputMetric(out_to_metric=lambda out, j=i: out.losses[j]).attach(engine, f"{LOSS_NAME}-{i}")

        if engine.state.aux_loss is not None:
            OutputMetric(out_to_metric=lambda out: out.aux_loss).attach(engine, AUX_LOSS_NAME)
            if len(engine.state.aux_loss) > 1:
                for i in range(len(engine.state.aux_loss)):
                    OutputMetric(out_to_metric=lambda out, j=i: out.aux_losses[j]).attach(
                        engine, f"{AUX_LOSS_NAME}-{i}"
                    )
