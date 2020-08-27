from __future__ import annotations

import typing
from collections import OrderedDict
from time import perf_counter

import ignite.engine
import torch

if typing.TYPE_CHECKING:
    from hylfm.setup.base import EvalStage, TrainStage


def step(engine: ignite.engine.Engine, tensors: typing.OrderedDict[str, typing.Any], train: bool):
    stage: typing.Union[EvalStage, TrainStage] = engine.state.stage
    model: torch.nn.Module = engine.state.model

    tensors = stage.batch_preprocessing_in_step(tensors)
    model.train(train)

    start = perf_counter()
    tensors = model(tensors)
    engine.state.compute_time += perf_counter() - start
    tensors = engine.state.batch_postprocessing(tensors)

    if train:
        tensors = engine.state.criterion(tensors)
        loss = tensors[stage.criterion_setup.name] / stage.batch_multiplier
        loss.backward()
        if (engine.state.iteration + 1) % stage.batch_multiplier == 0:
            engine.state.optimizer.step()
            engine.state.optimizer.zero_grad()

    for bmeta in tensors["meta"]:
        bmeta["log_dir"] = stage.log_path / f"ds{'-'.join([str(bidx) for bidx in bmeta['dataset_idx']])}"
        bmeta["log_dir"].mkdir(exist_ok=True, parents=True)
        for tensor_name in tensors:
            if tensor_name == "meta" or tensor_name not in bmeta:
                continue

            tmeta = bmeta[tensor_name]
            tmeta["log_dir"] = bmeta["log_dir"] / tensor_name
            tmeta["log_dir"].mkdir(exist_ok=True, parents=True)

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
    with torch.no_grad():
        return step(engine=engine, tensors=tensors, train=False)
