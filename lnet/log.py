from __future__ import annotations

import logging
import typing
from collections import OrderedDict
from concurrent.futures.thread import ThreadPoolExecutor

import numpy
import torch
import torch.utils.tensorboard
import yaml
from ignite.engine import Engine, Events
from tifffile import imsave
from tqdm import tqdm

from lnet import settings
from lnet.transforms import LightFieldFromChannel
from lnet.utils.plotting import get_batch_figure

if typing.TYPE_CHECKING:
    from lnet.setup import Stage
    from lnet.setup.base import EvalStage, TrainStage

logger = logging.getLogger(__name__)


def save_output(engine: Engine):
    tensors: typing.OrderedDict[str, typing.Any] = engine.state.output
    assert isinstance(tensors, OrderedDict)

    stage: Stage = engine.state.stage
    tensors_to_save = stage.outputs_to_save
    if tensors_to_save is None:
        tensors_to_save = list(tensors.keys())

    def save_tensor(name: str, tensor: typing.Any, meta: dict):
        save_to = stage.log_path / "output" / str(meta["idx"])
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

    with ThreadPoolExecutor(max_workers=settings.max_workers_save_output) as executor:
        for tensor_name in tensors_to_save:
            for tensor, meta in zip(tensors[tensor_name], tensors["meta"]):
                executor.submit(save_tensor, tensor_name, tensor, meta)


class BaseLogger:
    def __init__(self, *, stage: Stage, input: str, prediction: str, target: str, voxel_losses: typing.List[str]):
        self.stage = stage
        self.input = input
        self.prediction = prediction
        self.target = target
        self.voxel_losses = voxel_losses
        self._writer = None
        self.lightfield_from_channel = LightFieldFromChannel(nnum=stage.setup.nnum)

    def log_scalars(self, engine: Engine) -> None:
        pass

    def log_images(self, engine: Engine) -> None:
        pass


class TqdmLogger(BaseLogger):
    _pbar = None
    # _is_registered: bool = False
    _last_it: int = 0

    @property
    def pbar(self):
        if self._pbar is None:
            self._pbar = tqdm(
                self.stage.epoch_length, dynamic_ncols=True, unit="epoch", unit_divisor=self.stage.epoch_length
            )
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


# for i in range(10):
#     sleep(0.1)
#     pbar.update(10)
# pbar.close()


class FileLogger(BaseLogger):
    def _log_metric(self, engine: Engine, name: str):
        metric_log_file = self.stage.log_path / f"{name}.txt"
        with metric_log_file.open(mode="a") as file:
            file.write(f"{engine.state.iteration}\t{engine.state.metrics[name]}\n")

    def log_scalars(self, engine: Engine):
        [self._log_metric(engine, name) for name in engine.state.metrics]


class TensorBoadLogger(BaseLogger):
    @property
    def writer(self):
        if self._writer is None:
            self._writer = torch.utils.tensorboard.SummaryWriter(str(self.stage.log_path))

        return self._writer

    def log_scalars(self, engine: Engine):
        iteration = engine.state.iteration
        for k, v in engine.state.metrics.items():
            self.writer.add_scalar(tag=f"{self.stage.name}/{k}", scalar_value=v, global_step=iteration)

    def log_images(self, engine: Engine):
        iteration = engine.state.iteration
        output = engine.state.output


        ipt_batch = output[self.input]
        if isinstance(ipt_batch, torch.Tensor):
            ipt_batch = ipt_batch.cpu().numpy()

        pred_batch = output[self.prediction].detach().cpu().numpy()
        tgt_batch = output[self.target].cpu().numpy()
        voxel_losses = [output[vl].detach().cpu().numpy() for vl in self.voxel_losses]

        fig = get_batch_figure(
            ipt_batch=ipt_batch, pred_batch=pred_batch, tgt_batch=tgt_batch, voxel_losses=voxel_losses
        )

        self.writer.add_figure(tag=f"{self.stage.name}/batch", figure=fig, global_step=iteration)


class TensorBoadEvalLogger(TensorBoadLogger):
    def __init__(self, *, stage: EvalStage, **super_kwargs):
        super().__init__(stage=stage, **super_kwargs)


class TensorBoadTrainLogger(TensorBoadLogger):
    def __init__(self, *, stage: TrainStage, **super_kwargs):
        super().__init__(stage=stage, **super_kwargs)


class MultiLogger(BaseLogger):
    def __init__(self, loggers: typing.List[str], **super_kwargs):
        super().__init__(**super_kwargs)
        self.loggers = [globals().get(lgr)(**super_kwargs) for lgr in loggers]

    def log_scalars(self, engine: Engine) -> None:
        [lgr.log_scalars(engine=engine) for lgr in self.loggers]

    def log_images(self, engine: Engine) -> None:
        [lgr.log_images(engine=engine) for lgr in self.loggers]


class MultiEvalLogger(MultiLogger):
    def __init__(self, **super_kwargs):
        super().__init__(
            loggers=[TqdmLogger.__name__, FileLogger.__name__, TensorBoadEvalLogger.__name__], **super_kwargs
        )


class MultiTrainLogger(MultiLogger):
    def __init__(self, **super_kwargs):
        super().__init__(
            loggers=[TqdmLogger.__name__, FileLogger.__name__, TensorBoadTrainLogger.__name__], **super_kwargs
        )
