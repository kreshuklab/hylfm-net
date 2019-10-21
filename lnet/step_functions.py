from functools import partial
from typing import Union, Callable, Any

import torch
from ignite.utils import convert_tensor

from time import perf_counter

from lnet.engine import TrainEngine, EvalEngine
from lnet.output import Output


def step(engine: Union[EvalEngine, TrainEngine], batch, train: bool):
    start = perf_counter()

    model = engine.model
    device = next(model.parameters()).device

    if train:
        model.train()
        optimizer = engine.state.optimizer
        optimizer.zero_grad()
    else:
        model.eval()

    has_aux = len(batch) == 3
    if has_aux:
        ipt, tgt, aux_tgt = batch
        aux_tgt = convert_tensor(aux_tgt, device=device, non_blocking=False)
    else:
        ipt, tgt = batch
        aux_tgt = None

    ipt = convert_tensor(ipt, device=device, non_blocking=False)
    tgt = convert_tensor(tgt, device=device, non_blocking=False)
    pred = model(ipt)
    if has_aux:
        pred, aux_pred = pred
        aux_losses = [w * lf(aux_pred, tgt) for w, lf in engine.stae.aux_loss]
        aux_loss = sum(aux_losses)
    else:
        aux_pred = None
        aux_losses = None
        aux_loss = None

    losses = [w * lf(pred, tgt) for w, lf in engine.state.loss]
    total_loss = sum(losses)
    loss = total_loss
    if has_aux:
        total_loss += aux_loss

    if train:
        total_loss.backward()
        optimizer.step()

    engine.state.compute_time += perf_counter() - start
    return Output(
        ipt=ipt,
        tgt=tgt,
        aux_tgt=aux_tgt,
        pred=pred,
        aux_pred=aux_pred,
        loss=loss,
        aux_loss=aux_loss,
        losses=losses,
        aux_losses=aux_losses,
    )


training_step: Callable[[TrainEngine, Any], Output] = partial(step, train=True)
inference_step: Callable[[EvalEngine, Any], Output] = torch.no_grad()(partial(step, train=False))
