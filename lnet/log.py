from __future__ import annotations

import logging
import time
import typing
from collections import OrderedDict
from concurrent.futures.thread import ThreadPoolExecutor
from pathlib import Path

import numpy
import torch
import torch.utils.tensorboard
import yaml
from ignite.engine import Engine, Events
from tifffile import imsave
from tqdm import tqdm

from lnet import settings
from lnet.utils import PeriodUnit, Period
from lnet.utils.plotting import get_batch_figure

if typing.TYPE_CHECKING:
    from lnet.setup import Stage

logger = logging.getLogger(__name__)


class BaseLogger:
    def __init__(self, *, stage: Stage, tensor_names: typing.Optional[typing.Set[str]]):
        self.stage = stage
        self.tensor_names = tensor_names

    def log_scalars(self, engine: Engine) -> None:
        pass

    def get_tensor_names(self, engine: Engine) -> typing.Set[str]:
        if self.tensor_names is None:
            tensors: typing.OrderedDict[str, typing.Any] = engine.state.output
            assert isinstance(tensors, OrderedDict)
            tensor_names = {tn for tn in tensors.keys() if tn != "meta"}
        else:
            tensor_names = self.tensor_names

        return tensor_names

    def log_tensors(self, engine: Engine) -> None:
        pass

    def shutdown(self) -> None:
        pass

    def register_callbacks(self, engine: Engine, scalars_every: Period, tensors_every: Period) -> None:
        if scalars_every.unit == PeriodUnit.epoch:
            event = Events.EPOCH_COMPLETED
        elif scalars_every.unit == PeriodUnit.iteration:
            event = Events.ITERATION_COMPLETED
        else:
            raise NotImplementedError

        engine.add_event_handler(event(every=scalars_every.value), self.log_scalars)

        if tensors_every.unit == PeriodUnit.epoch:
            event = Events.EPOCH_COMPLETED
        elif tensors_every.unit == PeriodUnit.iteration:
            event = Events.ITERATION_COMPLETED
        else:
            raise NotImplementedError

        engine.add_event_handler(event(every=tensors_every.value), self.log_tensors)


class TqdmLogger(BaseLogger):
    _pbar = None
    # _is_registered: bool = False
    _last_it: int = 0

    @property
    def pbar(self):
        if self._pbar is None:
            self._pbar = tqdm(self.stage.epoch_length)
            # self._pbar = tqdm(
            #     self.stage.epoch_length, dynamic_ncols=True, unit="epoch", unit_divisor=self.stage.epoch_length
            # )
            # assert len(self._pbar) == self.stage.epoch_length
            # if not self._is_registered:
            #     self._is_registered = True
            #     def reset_pbar(engine: Engine, pbar=self._pbar):
            #         pbar.reset()
            #
            #     self.stage.engine.add_event_handler(Events.EPOCH_COMPLETED, reset_pbar)
            #
            #     def close_pbar(engine: Engine, pbar=self._pbar):
            #         pbar.close()
            #
            #     self.stage.engine.add_event_handler(Events.EPOCH_COMPLETED, close_pbar)

        return self._pbar

    def log_scalars(self, engine: Engine) -> None:
        it = engine.state.iteration
        its_passed = it - self._last_it
        self._last_it = it
        self.pbar.update(its_passed)

    def shutdown(self) -> None:
        self.pbar.close()
        super().shutdown()


class FileLogger(BaseLogger):
    def _log_metric(self, engine: Engine, name: str):
        metric_log_file = self.stage.log_path / f"{name}.txt"
        with metric_log_file.open(mode="a") as file:
            file.write(f"{engine.state.iteration}\t{engine.state.metrics[name]}\n")

    def log_scalars(self, engine: Engine):
        [self._log_metric(engine, name) for name in engine.state.metrics]

    def log_tensors(self, engine: Engine):
        tensor_names = self.get_tensor_names(engine)
        tensors: typing.OrderedDict[str, typing.Any] = engine.state.output

        with ThreadPoolExecutor(max_workers=settings.max_workers_file_logger) as executor:
            for tn in tensor_names:
                for tensor, meta in zip(tensors[tn], tensors["meta"]):
                    executor.submit(self._save_tensor, tn, tensor, meta, self.stage.log_path)

    @staticmethod
    def _save_tensor(name: str, tensor: typing.Any, meta: dict, log_path: Path):
        save_to = log_path / "output" / str(meta["idx"])
        save_to.mkdir(parents=True, exist_ok=True)
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.detach().cpu().numpy()

        save_to = save_to / name
        if isinstance(tensor, numpy.ndarray):
            try:
                imsave(str(save_to.with_suffix(".tif")), tensor, compress=2)
            except Exception as e:
                logger.error(e, exc_info=True)
                imsave(str(save_to.with_suffix(".tif")), tensor, compress=2, bigtiff=True)

        elif isinstance(tensor, dict):
            with save_to.with_suffix(".yml").open("w") as f:
                yaml.dump(tensor, f)
        else:
            raise NotImplementedError(type(tensor))


class TensorBoardLogger(BaseLogger):
    _writer = None

    @property
    def writer(self):
        if self._writer is None:
            self._writer = torch.utils.tensorboard.SummaryWriter(str(self.stage.log_path))

        return self._writer

    def log_scalars(self, engine: Engine):
        super().log_scalars(engine)
        iteration = engine.state.iteration
        for k, v in engine.state.metrics.items():
            self.writer.add_scalar(tag=f"{self.stage.name}/{k}", scalar_value=v, global_step=iteration)

    def log_tensors(self, engine: Engine):
        tensor_names = self.get_tensor_names(engine)
        iteration = engine.state.iteration
        output = engine.state.output

        tensors = OrderedDict(
            [
                (tn, output[tn].detach().cpu().numpy()) if isinstance(output[tn], torch.Tensor) else (tn, output[tn])
                for tn in tensor_names
            ]
        )

        fig = get_batch_figure(tensors=tensors, return_array=False)
        # fig_array = get_batch_figure(tensors=tensors, return_array=True)

        self.writer.add_figure(tag=f"{self.stage.name}/batch", figure=fig, global_step=iteration)
        # self.writer.add_image(tag=f"{self.stage.name}/batch", img_tensor=fig_array, global_step=iteration, dataformats="HWC")
        # self.writer.add_image(tag=f"{self.stage.name}/batch", img_tensor=fig_array[..., :3].astype("float") / 255, global_step=iteration, dataformats="HWC")

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
