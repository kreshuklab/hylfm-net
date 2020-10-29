import logging
from typing import Union

import numpy
import torch
from ignite.engine import Engine, Events

from hylfm.losses import LossOnTensorsTorchMixin

logger = logging.getLogger(__name__)


class WeightedLossOnTensorsTorchMixin(LossOnTensorsTorchMixin):
    def __init__(
        self,
        *super_args,
        engine: Engine,
        threshold: Union[float, str],
        initial_weight: float,
        decay_by: float,
        every_nth_epoch: int,
        apply_below_threshold: bool = False,
        **super_kwargs,
    ):
        super().__init__(*super_args, reduction="none", **super_kwargs)
        if isinstance(threshold, float):
            self.threshold = threshold
            self.percentile = None
        else:
            assert isinstance(threshold, str) and "percentile" in threshold
            self.percentile = float(threshold.replace("percentile", ""))
            self.threshold = None

        self.apply_below_threshold = apply_below_threshold

        self.weight = initial_weight - 1.0

        @engine.on(Events.EPOCH_COMPLETED)
        def decay_weight(engine):
            if engine.state.epoch % every_nth_epoch == 0:
                self.weight *= decay_by
                logger.info("decayed loss weight to %f (+1.0)", self.weight)

    def forward(self, prediction, target):
        loss = super().forward(prediction, target)  # noqa

        if self.training:  # noqa
            if self.threshold is None:
                threshold = numpy.percentile(target, q=self.percentile)
            else:
                threshold = self.threshold

            if self.apply_below_threshold:
                mask = target < threshold
            else:
                mask = target >= threshold

            loss_additional_weights = torch.zeros_like(loss)
            loss_additional_weights[mask] = loss[mask] * self.weight
            loss = loss + loss_additional_weights

        return {self.name: loss.mean(), self.name + "_pixelwise": loss}
