from __future__ import annotations

import logging
import time
import typing
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy
import torch
import torch.utils.tensorboard
import yaml
from ignite.engine import Engine, Events
from tifffile import imsave as imwrite
from tqdm import tqdm

from lnet import settings
from lnet.utils import PeriodUnit, Period
from lnet.utils.plotting import get_batch_figure

if typing.TYPE_CHECKING:
    from lnet.setup import Stage

logger = logging.getLogger(__name__)


def log_exception(func):
    def wrap(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except Exception as e:
            logger.error(e, exc_info=True)

    return wrap


class BaseLogger:
    def __init__(self, *, stage: Stage, tensor_names: typing.Optional[typing.Sequence[str]]):
        self.stage = stage
        self.tensor_names = tensor_names

    def get_unit_and_step(self, engine: Engine, event: Events):
        if event == Events.ITERATION_COMPLETED:
            step = engine.state.iteration + self.stage.run_count * self.stage.max_epochs * engine.state.epoch_length
            unit = "it"
        elif event == Events.EPOCH_COMPLETED:
            step = engine.state.epoch + self.stage.run_count * self.stage.max_epochs
            unit = "ep"
        else:
            raise NotImplementedError(event)

        return unit, step

    def log_scalars(self, engine: Engine) -> typing.Tuple[str, int]:
        for name, metric in self.stage.metric_instances.items():
            if name in self.stage.metrics:
                metric.completed(engine=engine, name=name)

        return self.get_unit_and_step(engine, self.scalar_event)

    def log_tensors(self, engine: Engine) -> typing.Tuple[str, int]:
        return self.get_unit_and_step(engine, self.tensor_event)

    def get_tensor_names(self, engine: Engine) -> typing.Sequence[str]:
        if self.tensor_names is None:
            tensors: typing.OrderedDict[str, typing.Any] = engine.state.output
            assert isinstance(tensors, OrderedDict)
            tensor_names = [tn for tn in tensors if tn != "meta"]
        else:
            tensor_names = self.tensor_names

        return tensor_names

    def shutdown(self) -> None:
        pass

    def register_callbacks(
        self,
        engine: Engine,
        scalars_every: typing.Optional[Period] = None,
        tensors_every: typing.Optional[Period] = None,
    ) -> None:
        if scalars_every is not None:
            if scalars_every.unit == PeriodUnit.epoch:
                self.scalar_event = Events.EPOCH_COMPLETED
            elif scalars_every.unit == PeriodUnit.iteration:
                self.scalar_event = Events.ITERATION_COMPLETED
            else:
                raise NotImplementedError

            engine.add_event_handler(self.scalar_event(every=scalars_every.value), self.log_scalars)

        if tensors_every is not None:
            if tensors_every.unit == PeriodUnit.epoch:
                self.tensor_event = Events.EPOCH_COMPLETED
            elif tensors_every.unit == PeriodUnit.iteration:
                self.tensor_event = Events.ITERATION_COMPLETED
            else:
                raise NotImplementedError

            engine.add_event_handler(self.tensor_event(every=tensors_every.value), self.log_tensors)


class TqdmLogger(BaseLogger):
    _pbar = None
    _last_it: int = 0

    def get_pbar(self, engine: Engine):
        if self._pbar is None:
            self._pbar = tqdm(total=self.stage.epoch_length)
            self._pbar.set_description(f"epoch {self.stage.run_count * self.stage.max_epochs + engine.state.epoch}")

        return self._pbar

    def register_callbacks(
        self,
        engine: Engine,
        scalars_every: typing.Optional[Period] = None,
        tensors_every: typing.Optional[Period] = None,
    ) -> None:
        if scalars_every is None:
            scalars_every = Period(value=1, unit="iteration")

        if scalars_every.unit == PeriodUnit.iteration:
            self.update_event = Events.ITERATION_COMPLETED
        else:
            raise NotImplementedError

        engine.add_event_handler(self.update_event(every=scalars_every.value), self.update_progress)
        engine.add_event_handler(Events.EPOCH_COMPLETED, self.reset_progress)

    def update_progress(self, engine: Engine) -> None:
        it = engine.state.iteration
        its_passed = it - self._last_it
        self._last_it = it
        self.get_pbar(engine).update(its_passed)

    def reset_progress(self, _: typing.Optional[Engine] = None) -> None:
        if self._pbar is not None:
            self._pbar.close()
            self._pbar = None

    def shutdown(self) -> None:
        self.reset_progress()
        super().shutdown()


class FileLogger(BaseLogger):
    def _log_metric(self, engine: Engine, name: str, unit: str, step: int):
        metric_log_file = self.stage.log_path / f"{name}.txt"
        with metric_log_file.open(mode="a") as file:
            file.write(f"{unit}\t{step}\t{engine.state.metrics[name]}\n")

    @log_exception
    def log_scalars(self, engine: Engine):
        unit, step = super().log_scalars(engine)
        [self._log_metric(engine, name, unit, step) for name in engine.state.metrics]
        return unit, step

    @log_exception
    def log_tensors(self, engine: Engine):
        unit, step = super().log_tensors(engine)
        tensor_names = self.get_tensor_names(engine)
        tensors: typing.OrderedDict[str, typing.Any] = engine.state.output

        with ThreadPoolExecutor(max_workers=settings.max_workers_file_logger) as executor:
            futs = []
            for tn in tensor_names:
                for tensor, tmeta in zip(tensors[tn], tensors["meta"]):
                    futs.append(executor.submit(self._save_tensor, tn, tensor, tmeta["log_path"]))
            for fut in as_completed(futs):
                e = fut.exception()
                if e is not None:
                    logger.error(e, exc_info=True)

        return unit, step

    @staticmethod
    def _save_tensor(name: str, tensor: typing.Any, folder: Path):
        assert len(tensor.shape) in (3, 4), f"expected tensor without batch dim, but got {tensor.shape}"
        folder.mkdir(parents=True, exist_ok=True)
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.detach().cpu().numpy()

        tensor = numpy.moveaxis(tensor, 0, -1)  # save channel last
        dtype_map = {"float64": "float32"}
        tensor = tensor.astype(dtype_map.get(str(tensor.dtype), tensor.dtype))

        folder = folder / name
        if isinstance(tensor, numpy.ndarray):
            try:
                imwrite(str(folder.with_suffix(".tif")), tensor, compress=2)
            except Exception as e:
                logger.error(e, exc_info=True)
                imwrite(str(folder.with_suffix(".tif")), tensor, compress=2, bigtiff=True)

        elif isinstance(tensor, dict):
            with folder.with_suffix(".yml").open("w") as f:
                yaml.dump(tensor, f)
        else:
            raise NotImplementedError(type(tensor))


class TensorBoardLogger(BaseLogger):
    _writer = None

    @property
    def writer(self):
        if self._writer is None:
            self._writer = torch.utils.tensorboard.SummaryWriter(str(self.stage.log_path.parent.parent))

        return self._writer

    @log_exception
    def log_scalars(self, engine: Engine):
        unit, step = super().log_scalars(engine)
        for k, v in engine.state.metrics.items():
            self.writer.add_scalar(tag=f"{self.stage.name}-{unit}/{k}", scalar_value=v, global_step=step)

        return unit, step

    @log_exception
    def log_tensors(self, engine: Engine):
        unit, step = super().log_tensors(engine)

        tensor_names = self.get_tensor_names(engine)
        output = engine.state.output

        tensors = OrderedDict(
            [
                (tn, output[tn].detach().cpu().numpy()) if isinstance(output[tn], torch.Tensor) else (tn, output[tn])
                for tn in tensor_names
            ]
        )

        fig = get_batch_figure(tensors=tensors, return_array=False)
        # fig_array = get_batch_figure(tensors=tensors, return_array=True)

        self.writer.add_figure(tag=f"{self.stage.name}-{unit}/batch", figure=fig, global_step=step)
        # self.writer.add_image(tag=f"{self.stage.name}/batch", img_tensor=fig_array, global_step=iteration, dataformats="HWC")
        # self.writer.add_image(tag=f"{self.stage.name}/batch", img_tensor=fig_array[..., :3].astype("float") / 255, global_step=iteration, dataformats="HWC")
        return unit, step

    def shutdown(self):
        if self._writer is not None:
            time.sleep(0.1)  # make sure everything is written
            self._writer.close()

        super().shutdown()


class MultiLogger(BaseLogger):
    def __init__(self, loggers: typing.List[str], **super_kwargs):
        super().__init__(**super_kwargs)
        self.loggers = [globals().get(lgr)(**super_kwargs) for lgr in loggers]

    def log_scalars(self, engine: Engine) -> None:
        [lgr.log_scalars(engine=engine) for lgr in self.loggers]

    def log_tensors(self, engine: Engine) -> None:
        [lgr.log_tensors(engine=engine) for lgr in self.loggers]

    def shutdown(self) -> None:
        [lgr.shutdown() for lgr in self.loggers]
