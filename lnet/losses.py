from typing import Union

import torch
from ignite.engine import Events
from inferno.extensions.criteria import SorensenDiceLoss

from lnet.engine import TrainEngine, EvalEngine


class WeightedL1Loss(torch.nn.L1Loss):
    def __init__(
        self,
        engine: Union[EvalEngine, TrainEngine],
        threshold: float,
        initial_weight: float,
        decay_by: float,
        every_nth_epoch: int,
    ):
        super().__init__(reduction="none")

        if isinstance(engine, TrainEngine):
            self.threshold = threshold
            self.weight = initial_weight

            @engine.on(Events.EPOCH_COMPLETED)
            def decay_weight(engine):
                if engine.state.epoch % every_nth_epoch == 0:
                    self.weight = (self.weight - 1.0) * decay_by + 1.0

    def forward(self, input, target):
        l1 = super().forward(input, target)
        if self.taining:
            l1[target > self.threshold] *= self.weight

        return l1.mean()


known_losses = {
    "BCEWithLogitsLoss": lambda engine: [(1.0, torch.nn.BCEWithLogitsLoss())],
    "SorensenDiceLoss": lambda engine: [(1.0, SorensenDiceLoss(channelwise=False, eps=1.0e-4))],
    "WeightedL1Loss": lambda **kwargs: [(1.0, WeightedL1Loss(**kwargs))],
}
