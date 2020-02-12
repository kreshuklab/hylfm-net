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

    z_slices = None
    if len(batch[-1].shape) == 1:
        z_slices = batch[-1]
        batch = batch[:-1]

    has_aux = len(batch) == 3
    if has_aux:
        if isinstance(batch, (list, tuple)):
            ipt, tgt, aux_tgt = batch
            ipt = convert_tensor(ipt, device=device, non_blocking=True)
            tgt = convert_tensor(tgt, device=device, non_blocking=True)
            aux_tgt = convert_tensor(aux_tgt, device=device, non_blocking=True)
        else:
            ipt = batch
            ipt = convert_tensor(ipt, device=device, non_blocking=True)
            tgt = None
            aux_tgt = None
    else:
        if isinstance(batch, (list, tuple)):
            ipt, tgt = batch
            ipt = convert_tensor(ipt, device=device, non_blocking=True)
            tgt = convert_tensor(tgt, device=device, non_blocking=True)
        else:
            ipt = batch
            ipt = convert_tensor(ipt, device=device, non_blocking=True)
            tgt = None

    if z_slices is None:
        pred = model(ipt)
    else:
        pred = model(ipt, z_slices=z_slices)

    if has_aux:
        pred, aux_pred = pred
        if tgt is not None:
            aux_losses = [w * lf(aux_pred, tgt) for w, lf in engine.state.aux_loss]
            aux_loss = sum(aux_losses)

    if tgt is None or not engine.state.loss:
        losses = []
        voxel_losses = []
        loss = torch.FloatTensor((0.0,))
        if tgt is None:
            tgt = torch.FloatTensor((0.0,))
    else:
        raw_losses = [(w, lf(pred, tgt)) for w, lf in engine.state.loss]
        losses = [w * rl[0] for w, rl in raw_losses]
        voxel_losses = [w * rl[1] for w, rl in raw_losses if rl[1] is not None]
        loss = sum(losses)

    if has_aux:
        total_loss = loss + aux_loss
    else:
        total_loss = loss

    if train:
        total_loss.backward()
        optimizer.step()

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
