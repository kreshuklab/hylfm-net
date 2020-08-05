import logging
from typing import Tuple, Union

import numpy

from lnet.utils.detect_beads import match_beads
from .base import Metric

logger = logging.getLogger(__name__)


class BeadPrecisionRecall(Metric):
    def __init__(
        self,
        *,
        dist_threshold: float,
        scaling: Tuple[float, float, float],
        min_sigma: float,
        max_sigma: float,
        sigma_ratio: float,
        threshold: float,
        overlap: float,
        exclude_border: Union[Tuple[int, ...], int, bool],
        pred: str = "pred",
        tgt: str = "tgt",
        meta: str = "meta",
        **super_kwargs,
    ):
        super().__init__(pred=pred, tgt=tgt, meta=meta, **super_kwargs)
        self.pred_name = pred
        self.match_beads_kwargs = {
            "min_sigma": min_sigma,
            "max_sigma": max_sigma,
            "sigma_ratio": sigma_ratio,
            "threshold": threshold,
            "overlap": overlap,
            "exclude_border": exclude_border,
            "dist_threshold": dist_threshold,
            "scaling": scaling,
        }

    def reset(self):
        self.found_missing_extra = []

    def update_impl(self, *, pred, tgt, meta):
        try:
            btgt_idx, bpred_idx, fme, bead_pos_btgt, bead_pos_bpred = match_beads(
                tgt.detach().cpu().numpy(), pred.detach().cpu().numpy(), **self.match_beads_kwargs
            )
        except Exception as e:
            logger.warning("could not match beads, due to exception %s", e)
            logger.warning(e, exc_info=True)
        else:
            self.found_missing_extra += fme
            try:
                for tgt_idx, pred_idx, bead_pos_tgt, bead_pos_pred, smeta in zip(
                    btgt_idx, bpred_idx, bead_pos_btgt, bead_pos_bpred, meta
                ):
                    log_path = smeta[self.pred_name].get("log_path", smeta.get("log_path", None))
                    assert log_path is not None
                    sample_idx = smeta["idx"]
                    numpy.savetxt(
                        str(log_path / f"{sample_idx:05}_matched_tgt_beads.txt"),
                        tgt_idx,
                        fmt="%3i",
                        delimiter="\t",
                        newline="\n",
                    )
                    numpy.savetxt(
                        str(log_path / f"{sample_idx:05}_matched_pred_beads.txt"),
                        pred_idx,
                        fmt="%3i",
                        delimiter="\t",
                        newline="\n",
                    )
                    numpy.savetxt(
                        str(log_path / f"{sample_idx:05}_tgt_bead_pos.txt"),
                        bead_pos_tgt,
                        fmt="%3i",
                        delimiter="\t",
                        newline="\n",
                    )
                    numpy.savetxt(
                        str(log_path / f"{sample_idx:05}_pred_bead_pos.txt"),
                        bead_pos_pred,
                        fmt="%3i",
                        delimiter="\t",
                        newline="\n",
                    )
            except Exception as e:
                logger.error(e, exc_info=True)

    def compute_impl(self):
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

        return {"bead_precision": precision, "bead_recall": recall}
