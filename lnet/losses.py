import logging
import torch

from collections import namedtuple
from ignite.engine import Events
from inferno.extensions import criteria
from torch import nn, Tensor, FloatTensor
from typing import Union, Callable, Optional

from lnet.engine import TrainEngine, EvalEngine

logger = logging.getLogger(__name__)

Loss = namedtuple("Loss", ["reduced", "pixelwise"])


class WeightedL1Loss(nn.L1Loss):
    def __init__(
        self,
        engine: Union[EvalEngine, TrainEngine],
        threshold: float,
        initial_weight: float,
        decay_by: float,
        every_nth_epoch: int,
        apply_below_threshold: bool = False,
        inference_weight: Optional[float] = None,
    ):
        super().__init__(reduction="none")
        self.threshold = threshold
        self.apply_below_threshold = apply_below_threshold

        self.weight = initial_weight - 1.0
        if isinstance(engine, TrainEngine):
            self.instance_for_training = True

            @engine.on(Events.EPOCH_COMPLETED)
            def decay_weight(engine):
                if engine.state.epoch % every_nth_epoch == 0:
                    self.weight *= decay_by
                    logger.info("decayed loss weight to %f (+1.0)", self.weight)

        else:
            self.instance_for_training = False
            if inference_weight is not None:
                self.weight = inference_weight

    def forward(self, input, target):
        l1 = super().forward(input, target)

        if self.apply_below_threshold:
            mask = target < self.threshold
        else:
            mask = target >= self.threshold

        l1_additional_weights = torch.zeros_like(l1)
        l1_additional_weights[mask] = l1[mask] * self.weight
        l1 = l1 + l1_additional_weights

        return Loss(l1.mean(), l1)

    def train(self, mode=True):
        assert self.instance_for_training == mode
        super().train(mode=mode)


def reduced_loss_only_decorator(loss_forward: Callable[[Tensor, Tensor], Tensor]) -> Callable[[Tensor, Tensor], Loss]:
    def new_forward(self, input: Tensor, target: Tensor) -> Loss:
        loss = loss_forward(input, target)
        return Loss(loss, None)

    return new_forward


class BCEWithLogitsLoss(torch.nn.BCEWithLogitsLoss):
    def __init__(self, weight=None, pos_weight=None, **kwargs):
        if weight is not None:
            weight = FloatTensor(weight)

        if pos_weight is not None:
            pos_weight = FloatTensor(pos_weight)

        super().__init__(weight=weight, pos_weight=pos_weight, **kwargs)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        raise NotImplementedError

class SorensenDiceLoss(criteria.SorensenDiceLoss):
    def __init__(self, channelwise=False, **kwargs):
        super().__init__(channelwise=channelwise, **kwargs)

    def forward(self, input, target):
        loss = super().forward(input, target)
        return Loss(loss, None)


class MSELoss(torch.nn.MSELoss):
    def forward(self, input: Tensor, target: Tensor) -> Loss:
        loss = super().forward(input, target)
        return Loss(loss, None)


known_losses = {
    "BCEWithLogitsLoss": lambda engine, kwargs: [(1.0, BCEWithLogitsLoss(**kwargs))],
    "SorensenDiceLoss": lambda engine, kwargs: [(1.0, SorensenDiceLoss(**kwargs))],
    "WeightedL1Loss": lambda engine, kwargs: [(1.0, WeightedL1Loss(engine=engine, **kwargs))],
    "MSELoss": lambda engine, kwargs: [(1.0, MSELoss(**kwargs))],
}
