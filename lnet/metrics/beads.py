import logging
from typing import Callable

import numpy
from ignite.metrics import Metric

from lnet.utils.detect_beads import match_beads

logger = logging.getLogger(__name__)


class BeadPrecisionRecall(Metric):
    def __init__(self, output_to_tgt_pred: Callable = lambda out: (out.tgt, out.pred), dist_threshold: float = 5.0):
        super().__init__(output_transform=output_to_tgt_pred)
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
