from __future__ import annotations
import typing
from collections import OrderedDict
from time import perf_counter

import ignite.engine
import torch

if typing.TYPE_CHECKING:
    from lnet.setup import Setup


def step(engine: ignite.engine.Engine, tensors: typing.OrderedDict[str, typing.Any], train: bool):
    start = perf_counter()

    setup: Setup = engine.state.stage.setup
    model = setup.model


    model.train(train)
    if train:
        optimizer = engine.state.optimizer
        optimizer.zero_grad()
    else:
        optimizer = None

    # tensors = OrderedDict(
    #     [
    #         (name, (tensor.to(device=setup.device) if isinstance(tensor, torch.Tensor) else tensor))
    #         for name, tensor in tensors.items()
    #     ]
    # )
    tensors = model(tensors)

    if train:
        tensors = engine.state.criterion(tensors)
        loss = tensors[engine.state.criterion_name]
        loss.backward()
        optimizer.step()

    engine.state.compute_time += perf_counter() - start
    return tensors


def training_step(engine: ignite.engine.Engine, tensors: typing.OrderedDict) -> typing.OrderedDict:
    return step(engine=engine, tensors=tensors, train=True)


def inference_step(engine: ignite.engine.Engine, tensors: typing.OrderedDict) -> typing.OrderedDict:
    return step(engine=engine, tensors=tensors, train=False)
