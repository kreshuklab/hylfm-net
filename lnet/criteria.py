from __future__ import annotations

import logging
import typing
from collections import OrderedDict

import torch.nn
from ignite.engine import Engine, Events
from inferno.extensions import criteria
from torch import FloatTensor, Tensor

logger = logging.getLogger(__name__)


class CriterionWrapper(torch.nn.Module):
    def __init__(
        self, tensor_names: typing.Dict[str, str], criterion_class: torch.nn.Module, postfix: str = "", **kwargs
    ):
        super().__init__()
        self.input_kargs = tensor_names
        self.criterion = criterion_class(**kwargs)
        self.postfix = postfix

    def forward(self, tensors: typing.OrderedDict[str, typing.Any]):
        out = self.criterion.forward(**{name: tensors[tensor_name] for name, tensor_name in self.tensor_names.items()})
        if isinstance(out, OrderedDict):
            assert self.criterion.__class__.__name__ in out
            for loss_name, loss_value in out.items():
                loss_name += self.postfix
                assert loss_name not in tensors
                tensors[loss_name] = loss_value
        else:
            loss_name = self.criterion.__class__.__name__ + self.postfix
            assert loss_name not in tensors
            tensors[loss_name] = out

        return tensors


class WeightedL1Loss(torch.nn.L1Loss):
    def __init__(
        self,
        engine: Engine,
        threshold: float,
        initial_weight: float,
        decay_by: float,
        every_nth_epoch: int,
        apply_below_threshold: bool = False,
    ):
        super().__init__(reduction="none")
        self.threshold = threshold
        self.apply_below_threshold = apply_below_threshold

        self.weight = initial_weight - 1.0

        @engine.on(Events.EPOCH_COMPLETED)
        def decay_weight(engine):
            if engine.state.epoch % every_nth_epoch == 0:
                self.weight *= decay_by
                logger.info("decayed loss weight to %f (+1.0)", self.weight)

    def forward(self, input, target):
        l1 = super().forward(input, target)

        if self.apply_below_threshold:
            mask = target < self.threshold
        else:
            mask = target >= self.threshold

        if self.training:
            l1_additional_weights = torch.zeros_like(l1)
            l1_additional_weights[mask] = l1[mask] * self.weight
            l1 = l1 + l1_additional_weights

        return OrderedDict([(self.__class__.__name__, l1.mean()), (f"{self.__class__.__name__} pixelwise", l1)])


class WeightedSmoothL1Loss(torch.nn.SmoothL1Loss):
    def __init__(
        self,
        engine: Engine,
        threshold: float,
        initial_weight: float,
        decay_by: float,
        every_nth_epoch: int,
        apply_below_threshold: bool = False,
    ):
        super().__init__(reduction="none")
        self.threshold = threshold
        self.apply_below_threshold = apply_below_threshold

        self.weight = initial_weight - 1.0

        @engine.on(Events.EPOCH_COMPLETED)
        def decay_weight(engine):
            if engine.state.epoch % every_nth_epoch == 0:
                self.weight *= decay_by
                logger.info("decayed loss weight to %f (+1.0)", self.weight)

    def forward(self, input, target):
        l1 = super().forward(input, target)

        if self.training:
            if self.apply_below_threshold:
                mask = target < self.threshold
            else:
                mask = target >= self.threshold

            l1_additional_weights = torch.zeros_like(l1)
            l1_additional_weights[mask] = l1[mask] * self.weight
            l1 = l1 + l1_additional_weights

        return OrderedDict([(self.__class__.__name__, l1.mean()), (f"{self.__class__.__name__} pixelwise", l1)])


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
        return loss


class MSELoss(torch.nn.MSELoss):
    def forward(self, input: Tensor, target: Tensor):
        loss = super().forward(input, target)
        return loss
