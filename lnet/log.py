import logging
import typing
from collections import OrderedDict
from concurrent.futures.thread import ThreadPoolExecutor

import numpy
import torch
import yaml
from ignite.engine import Engine
from tifffile import imsave

from lnet import settings
from lnet.setup import Stage

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
