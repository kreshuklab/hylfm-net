from __future__ import annotations

import logging
import typing
from collections import OrderedDict
from concurrent.futures.thread import ThreadPoolExecutor

import numpy
import torch
import torch.utils.tensorboard
import yaml
from ignite.engine import Engine
from tifffile import imsave

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


class TensorBoadLogger:
    def __init__(self, *, stage: Stage, input: str, prediction: str, target: str, voxel_losses: typing.List[str]):
        self.stage = stage
        self.input = input
        self.prediction = prediction
        self.target = target
        self.voxel_losses = voxel_losses
        self._writer = None
        self.lightfield_from_channel = LightFieldFromChannel(nnum=stage.setup.nnum)
        self.logger = logging.getLogger(stage.name)

    @property
    def writer(self):
        if self._writer is None:
            self._writer = torch.utils.tensorboard.SummaryWriter(str(self.stage.log_path))

        return self._writer

    def log_scalars(self, engine: Engine):
        iteration = engine.state.iteration
        self.logger.info("%s Epoch: %4d Iteration: %6d", self.stage.name, engine.state.epoch, iteration)
        for k, v in engine.state.metrics.items():
            self.writer.add_scalar(tag=f"{self.stage.name}/{k}", scalar_value=v, global_step=iteration)

    def log_images(self, engine: Engine):
        iteration = engine.state.iteration
        output = engine.state.output
        ipt_batch = numpy.stack([self.lightfield_from_channel(ipt) for ipt in output[self.input].cpu().numpy()])

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
