from __future__ import annotations

import typing
from collections import OrderedDict
from time import perf_counter

import ignite.engine
import torch

if typing.TYPE_CHECKING:
    from lnet.setup.base import EvalStage, TrainStage


def step(engine: ignite.engine.Engine, tensors: typing.OrderedDict[str, typing.Any], train: bool):
    stage: typing.Union[EvalStage, TrainStage] = engine.state.stage
    model: torch.nn.Module = engine.state.model
    for tmeta in tensors["meta"]:
        tmeta["log_path"] = stage.log_path / f"ds{tmeta['dataset_idx']:02}" / f"{tmeta['idx']:05}"
        tmeta["log_path"].mkdir(exist_ok=True, parents=True)

    tensors = stage.batch_preprocessing_in_step(tensors)
    model.train(train)
    if train:
        optimizer = engine.state.optimizer
        optimizer.zero_grad()
    else:
        optimizer = None

    start = perf_counter()
    tensors = model(tensors)
    engine.state.compute_time += perf_counter() - start
    tensors = engine.state.batch_postprocessing(tensors)

    if train:
        tensors = engine.state.criterion(tensors)
        loss = tensors[stage.criterion_setup.name]
        loss.backward()
        optimizer.step()

    return OrderedDict(
        [
            (name, tensor.detach().to(device=torch.device("cpu"), non_blocking=True))
            if isinstance(tensor, torch.Tensor)
            else (name, tensor)
            for name, tensor in tensors.items()
        ]
    )


def training_step(engine: ignite.engine.Engine, tensors: typing.OrderedDict) -> typing.OrderedDict:
    return step(engine=engine, tensors=tensors, train=True)


def inference_step(engine: ignite.engine.Engine, tensors: typing.OrderedDict) -> typing.OrderedDict:
    return step(engine=engine, tensors=tensors, train=False)
