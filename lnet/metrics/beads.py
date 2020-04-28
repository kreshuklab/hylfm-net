import logging

import numpy
from ignite.metrics import Metric, MetricsLambda

from lnet.utils.detect_beads import match_beads
from ._utils import get_output_transform

logger = logging.getLogger(__name__)


class BeadPrecisionRecall(Metric):
    _required_output_keys = ("y_pred", "y", "meta")

    def __init__(self, *, dist_threshold: float = 5.0, **super_kwargs):
        super().__init__(**super_kwargs)
        self.dist_threshold = dist_threshold

    def reset(self):
        self.found_missing_extra = []

    def update(self, output):
        pred, tgt, meta = output
        try:
            btgt_idx, bpred_idx, fme, bead_pos_btgt, bead_pos_bpred = match_beads(
                tgt.detach().cpu().numpy(), pred.detach().cpu().numpy(), dist_threshold=self.dist_threshold
            )
        except Exception as e:
            logger.info(e)
        else:
            self.found_missing_extra += fme
            try:
                for tgt_idx, pred_idx, bead_pos_tgt, bead_pos_pred, tmeta in zip(
                    btgt_idx, bpred_idx, bead_pos_btgt, bead_pos_bpred, meta
                ):
                    log_path = tmeta["log_path"]
                    numpy.savetxt(
                        str(log_path / "matched_tgt_beads.txt"), tgt_idx, fmt="%3i", delimiter="\t", newline="\n"
                    )
                    numpy.savetxt(
                        str(log_path / "matched_pred_beads.txt"), pred_idx, fmt="%3i", delimiter="\t", newline="\n"
                    )
                    numpy.savetxt(
                        str(log_path / "tgt_bead_pos.txt"), bead_pos_tgt, fmt="%3i", delimiter="\t", newline="\n"
                    )
                    numpy.savetxt(
                        str(log_path / "pred_bead_pos.txt"), bead_pos_pred, fmt="%3i", delimiter="\t", newline="\n"
                    )
            except Exception as e:
                logger.error(e, exc_info=True)

    def compute(self):
        try:
            precision = numpy.asarray([f / (f + e) for f, m, e in self.found_missing_extra]).mean()
        except Exception as e:
            logger.info(e)
            precision = numpy.nan

        try:
            recall = numpy.asarray([f / (f + m) for f, m, e in self.found_missing_extra]).mean()
        except Exception as e:
            logger.info(e)
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
