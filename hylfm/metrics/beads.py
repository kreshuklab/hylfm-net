import collections
import logging
from typing import Any, DefaultDict, Dict, List, Optional, Tuple, Union

import numpy

from hylfm.detect_beads import match_beads
from .base import Metric

logger = logging.getLogger(__name__)


class BeadPrecisionRecall(Metric):
    minimize: bool = False
    found_missing_extra: list
    found_missing_extra_alongdim: DefaultDict[int, DefaultDict[int, List[Tuple[int, int, int]]]]
    max_shape: numpy.ndarray
    result_per_sample: List[Dict[str, Any]]

    def __init__(
        self,
        *,
        dist_threshold: float,
        scaling: Tuple[float, float, float],
        min_sigma: float,
        max_sigma: float,
        sigma_ratio: float,
        threshold: float,
        tgt_threshold: float,
        overlap: float,
        exclude_border: Union[Tuple[int, ...], int, bool],
        dim_names: Optional[str] = None,
        name: str = "{name}-{dim_name}",
    ):
        self.match_beads_kwargs = {
            "min_sigma": min_sigma,
            "max_sigma": max_sigma,
            "sigma_ratio": sigma_ratio,
            "threshold": threshold,
            "tgt_threshold": tgt_threshold,
            "overlap": overlap,
            "exclude_border": exclude_border,
            "dist_threshold": dist_threshold,
            "scaling": scaling,
        }
        super().__init__(name=name, dim_names="zyx" if dim_names is None else dim_names)

    def reset(self):
        self.found_missing_extra = []
        self.found_missing_extra_alongdim = collections.defaultdict(lambda: collections.defaultdict(list))
        self.max_shape = numpy.zeros(len(self.dim_names), dtype=numpy.int)

    def update_with_sample(self, *, prediction, target):
        # add batch dim = 1
        prediction = prediction[None]
        target = target[None]

        assert len(self.max_shape) == len(target.shape) - 2, "does init kwarg 'dim_names' have correct length?"
        self.max_shape = numpy.maximum(self.max_shape, target.shape[2:])
        try:
            btgt_idx, bpred_idx, fme, bead_pos_btgt, bead_pos_bpred = match_beads(
                target.detach().cpu().numpy(), prediction.detach().cpu().numpy(), **self.match_beads_kwargs
            )
        except Exception as e:
            logger.warning("could not match beads, due to exception %s", e)
            logger.warning(e, exc_info=True)
            return {}

        fme_alongdim = collections.defaultdict(lambda: collections.defaultdict(list))
        try:
            for tgt_idx, pred_idx, bead_pos_tgt, bead_pos_pred in zip(
                btgt_idx, bpred_idx, bead_pos_btgt, bead_pos_bpred
            ):
                if bead_pos_tgt.shape[0]:
                    for dim in range(bead_pos_tgt.shape[1]):
                        extra_in_pred = numpy.ones(bead_pos_pred.shape[0], numpy.bool)
                        extra_in_pred[pred_idx] = 0
                        for p in numpy.unique(bead_pos_tgt[:, dim]):
                            found_at_p: int = (bead_pos_tgt[tgt_idx, dim] == p).sum()
                            missing_at_p: int = (bead_pos_tgt[:, dim] == p).sum() - found_at_p
                            extra_at_p: int = (bead_pos_pred[extra_in_pred, dim] == p).sum()
                            fme_alongdim[dim][int(round(p))].append((found_at_p, missing_at_p, extra_at_p))

        except Exception as e:
            logger.error(e, exc_info=True)

        ret = self._compute(fme, fme_alongdim)

        self.found_missing_extra += fme
        for dim, fme_per_p in fme_alongdim.items():
            for p, fme_pp in fme_per_p.items():
                self.found_missing_extra_alongdim[dim][p] += fme_pp

        return ret

    @staticmethod
    def get_precision(found_missing_extra: List[Tuple[int, int, int]]):
        f, m, e = numpy.asarray(found_missing_extra).sum(axis=0)
        return float('nan') if e == 0 else float(f / (f + e))
        # return float(numpy.asarray([1.0 if e == 0 else f / (f + e) for f, m, e in found_missing_extra]).mean())

    @staticmethod
    def get_recall(found_missing_extra: List[Tuple[int, int, int]]):
        f, m, e = numpy.asarray(found_missing_extra).sum(axis=0)
        return float('nan') if m == 0 else float(f / (f + m))
        # return float(numpy.asarray([1.0 if m == 0 else f / (f + m) for f, m, e in found_missing_extra]).mean())

    def compute(self) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
        return self._compute(self.found_missing_extra, self.found_missing_extra_alongdim)

    def _compute(self, found_missing_extra, found_missing_extra_alongdim):
        ret = {
            "bead_precision": self.get_precision(found_missing_extra),
            "bead_recall": self.get_recall(found_missing_extra),
        }
        for dim, fme_per_p_dict in found_missing_extra_alongdim.items():

            fme_per_p = numpy.zeros((self.max_shape[dim], 3), dtype=float)
            for p, fme in fme_per_p_dict.items():
                fme_per_p[p] += numpy.asarray(fme).sum(axis=0)

            ret[f"bead_precision-{self.dim_names[dim]}"] = numpy.asarray(
                [1.0 if e == 0 else float(f / (f + e)) for f, m, e in fme_per_p]
            )
            ret[f"bead_recall-{self.dim_names[dim]}"] = numpy.asarray(
                [1.0 if m == 0 else float(f / (f + m)) for f, m, e in fme_per_p]
            )

        return ret
