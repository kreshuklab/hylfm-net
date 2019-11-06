from functools import partial
from typing import Union, Callable, Any

import torch
from ignite.utils import convert_tensor

from time import perf_counter

from lnet.engine import TrainEngine, EvalEngine
from lnet.output import Output, AuxOutput


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
        if isinstance(batch, (list, tuple)):
            ipt, tgt, aux_tgt = batch
            aux_tgt = convert_tensor(aux_tgt, device=device, non_blocking=False)
        else:
            ipt = batch
            tgt = None
            aux_tgt = None
    else:
        if isinstance(batch, (list, tuple)):
            ipt, tgt = batch
            tgt = convert_tensor(tgt, device=device, non_blocking=False)
        else:
            ipt = batch
            tgt = None

    ipt = convert_tensor(ipt, device=device, non_blocking=False)
    pred = model(ipt)
    if has_aux:
        pred, aux_pred = pred
        if tgt is not None:
            aux_losses = [w * lf(aux_pred, tgt) for w, lf in engine.state.aux_loss]
            aux_loss = sum(aux_losses)

    if tgt is not None:
        raw_losses = [(w, lf(pred, tgt)) for w, lf in engine.state.loss]
        losses = [w * rl[0] for w, rl in raw_losses]
        voxel_losses = [None if rl[1] is None else w * rl[1] for w, rl in raw_losses]
    else:
        losses = []
        voxel_losses = []

    loss = sum(losses)

    if has_aux:
        total_loss = loss + aux_loss
    else:
        total_loss = loss

    if train:
        total_loss.backward()
        optimizer.step()
    elif tgt is None:
        loss = torch.FloatTensor((0.0,))
        tgt = torch.FloatTensor((0.0,))

    engine.state.compute_time += perf_counter() - start
    if has_aux:
        # todo: get rid of AuxOutput and simply have tuples of tensors for tgt, pred, loss, etc...
        return AuxOutput(
            ipt=ipt,
            tgt=tgt,
            pred=pred,
            loss=loss,
            losses=losses,
            voxel_losses=voxel_losses,
            aux_tgt=aux_tgt,
            aux_pred=aux_pred,
            aux_loss=aux_loss,
            aux_losses=aux_losses,
        )
    else:
        return Output(ipt=ipt, tgt=tgt, pred=pred, loss=loss, losses=losses, voxel_losses=voxel_losses)


training_step: Callable[[TrainEngine, Any], Output] = partial(step, train=True)
inference_step: Callable[[EvalEngine, Any], Output] = torch.no_grad()(partial(step, train=False))
