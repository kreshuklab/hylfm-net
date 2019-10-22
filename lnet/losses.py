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

        self.threshold = threshold

        if isinstance(engine, TrainEngine):
            self.instance_for_training = True
            self.weight = initial_weight - 1.

            @engine.on(Events.EPOCH_COMPLETED)
            def decay_weight(engine):
                if engine.state.epoch % every_nth_epoch == 0:
                    self.weight *= decay_by

        else:
            self.instance_for_training = False
            self.weight = 1.0

    def forward(self, input, target):
        l1 = super().forward(input, target)
        if self.training:
            mask = target > self.threshold
            l1_additional_weights = torch.zeros_like(l1)
            l1_additional_weights[mask] = l1[mask] * self.weight
            l1 = l1 + l1_additional_weights

        return l1.mean()

    def train(self, mode=True):
        assert self.instance_for_training == mode
        super().train(mode=mode)


known_losses = {
    "BCEWithLogitsLoss": lambda engine, kwargs: [
        (
            1.0,
            torch.nn.BCEWithLogitsLoss(
                **{k: torch.FloatTensor(v) if k in ["weight", "pos_weight"] else v for k, v in kwargs.items()}
            ),
        )
    ],
    "SorensenDiceLoss": lambda engine, kwargs: [(1.0, SorensenDiceLoss(channelwise=kwargs.pop("channelwise", False)))],
    "WeightedL1Loss": lambda engine, kwargs: [(1.0, WeightedL1Loss(engine=engine, **kwargs))],
}
