import logging

import numpy
from ignite.metrics import Metric, MetricsLambda

from lnet.utils.detect_beads import match_beads
from ._utils import get_output_transform

logger = logging.getLogger(__name__)


class BeadPrecisionRecall(Metric):
    def __init__(self, *, dist_threshold: float = 5.0, **super_kwargs):
        super().__init__(**super_kwargs)
        self.dist_threshold = dist_threshold

    def reset(self):
        self.found_missing_extra = []

    def update(self, output):
        tgt, pred = output
        try:
            tgt_idx, pred_idx, fme = match_beads(
                tgt.detach().cpu().numpy(), pred.detach().cpu().numpy(), dist_threshold=self.dist_threshold
            )
            self.found_missing_extra += fme
        except Exception as e:
            logger.exception(e)

    def compute(self):
        try:
            precision = numpy.asarray([f / (f + e) for f, m, e in self.found_missing_extra]).mean()
        except Exception as e:
            logger.exception(e)
            precision = numpy.nan

        try:
            recall = numpy.asarray([f / (f + m) for f, m, e in self.found_missing_extra]).mean()
        except Exception as e:
            logger.exception(e)
            recall = numpy.nan

        return precision, recall


def get_BeadPrecision(*, initialized_metrics: dict, kwargs: dict):
    prec_and_recall = initialized_metrics.get(BeadPrecisionRecall.__name__, None)
    if prec_and_recall is None:
        initialized_metrics[BeadPrecisionRecall.__name__] = BeadPrecisionRecall(
            output_transform=get_output_transform(kwargs.pop("tensor_names")), **kwargs
        )

    class BeadPrecision(MetricsLambda):
        pass

    return BeadPrecision(lambda pr: pr[0], initialized_metrics[BeadPrecisionRecall.__name__])


def get_BeadRecall(*, initialized_metrics: dict, kwargs: dict):
    prec_and_recall = initialized_metrics.get(BeadPrecisionRecall.__name__, None)
    if prec_and_recall is None:
        initialized_metrics[BeadPrecisionRecall.__name__] = BeadPrecisionRecall(
            output_transform=get_output_transform(kwargs.pop("tensor_names")), **kwargs
        )

    class BeadRecall(MetricsLambda):
        pass

    return BeadRecall(lambda pr: pr[1], initialized_metrics[BeadPrecisionRecall.__name__])
