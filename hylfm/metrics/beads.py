import logging
import typing
from collections import defaultdict
from typing import Tuple, Union

import numpy

from hylfm.utils.detect_beads import match_beads
from .scale_minimize_vs import ScaleMinimizeVsMetric

logger = logging.getLogger(__name__)


class BeadPrecisionRecall(ScaleMinimizeVsMetric):
    def __init__(
        self,
        *super_args,
        dist_threshold: float,
        scaling: Tuple[float, float, float],
        min_sigma: float,
        max_sigma: float,
        sigma_ratio: float,
        threshold: float,
        tgt_threshold: float,
        overlap: float,
        exclude_border: Union[Tuple[int, ...], int, bool],
        tensor_names: typing.Optional[typing.Dict[str, str]] = None,
        dim_names: typing.Optional[str] = None,
        **super_kwargs,
    ):
        if tensor_names is None:
            tensor_names = {}

        if "pred" not in tensor_names:
            tensor_names["pred"] = "pred"

        if "tgt" not in tensor_names:
            tensor_names["tgt"] = "tgt"

        if "meta" not in tensor_names:
            tensor_names["meta"] = "meta"

        self.dim_names = "zyx" if dim_names is None else dim_names
        super().__init__(*super_args, tensor_names=tensor_names, **super_kwargs)
        self.pred_name = tensor_names["pred"]
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

    def reset(self):
        self.found_missing_extra = []
        self.found_missing_extra_alongdim = defaultdict(lambda: defaultdict(list))
        self.max_shape = numpy.zeros(len(self.dim_names), dtype=numpy.int)

    def update_impl(self, *, pred, tgt, meta):
        assert len(self.max_shape) == len(tgt.shape) - 2, "does init kwarg 'dim_names' have correct length?"
        self.max_shape = numpy.maximum(self.max_shape, tgt.shape[2:])
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
                    if bead_pos_tgt.shape[0]:
                        for dim in range(bead_pos_tgt.shape[1]):
                            extra_in_pred = numpy.ones(bead_pos_pred.shape[0], numpy.bool)
                            extra_in_pred[pred_idx] = 0
                            for p in numpy.unique(bead_pos_tgt[:, dim]):
                                found_at_p = (bead_pos_tgt[tgt_idx, dim] == p).sum()
                                missing_at_p = (bead_pos_tgt[:, dim] == p).sum() - found_at_p
                                extra_at_p = (bead_pos_pred[extra_in_pred, dim] == p).sum()
                                self.found_missing_extra_alongdim[dim][int(round(p))].append(
                                    (found_at_p, missing_at_p, extra_at_p)
                                )

                    # log_dir = smeta[self.pred_name].get("log_dir", smeta.get("log_dir", None))
                    # assert log_dir is not None
                    # sample_idx = smeta["idx"]
                    # numpy.savetxt(
                    #     str(log_dir / f"{sample_idx:05}_matched_tgt_beads.txt"),
                    #     tgt_idx,
                    #     fmt="%3i",
                    #     delimiter="\t",
                    #     newline="\n",
                    # )
                    # numpy.savetxt(
                    #     str(log_dir / f"{sample_idx:05}_matched_pred_beads.txt"),
                    #     pred_idx,
                    #     fmt="%3i",
                    #     delimiter="\t",
                    #     newline="\n",
                    # )
                    # numpy.savetxt(
                    #     str(log_dir / f"{sample_idx:05}_tgt_bead_pos.txt"),
                    #     bead_pos_tgt,
                    #     fmt="%3i",
                    #     delimiter="\t",
                    #     newline="\n",
                    # )
                    # numpy.savetxt(
                    #     str(log_dir / f"{sample_idx:05}_pred_bead_pos.txt"),
                    #     bead_pos_pred,
                    #     fmt="%3i",
                    #     delimiter="\t",
                    #     newline="\n",
                    # )
            except Exception as e:
                logger.error(e, exc_info=True)

    @staticmethod
    def get_precision(found_missing_extra: typing.List[typing.Tuple[int, int, int]]):
        f, m, e = numpy.asarray(found_missing_extra).sum(axis=0)
        return 1.0 if e == 0 else float(f / (f + e))
        # return float(numpy.asarray([1.0 if e == 0 else f / (f + e) for f, m, e in found_missing_extra]).mean())

    @staticmethod
    def get_recall(found_missing_extra: typing.List[typing.Tuple[int, int, int]]):
        f, m, e = numpy.asarray(found_missing_extra).sum(axis=0)
        return 1.0 if m == 0 else float(f / (f + m))
        # return float(numpy.asarray([1.0 if m == 0 else f / (f + m) for f, m, e in found_missing_extra]).mean())

    def compute_impl(self):
        ret = {
            "bead_precision": self.get_precision(self.found_missing_extra),
            "bead_recall": self.get_recall(self.found_missing_extra),
        }
        for dim, fme_per_p_dict in self.found_missing_extra_alongdim.items():
            # f = m = e = 0.0
            # counts_along_dim = numpy.zeros(self.max_shape[dim], dtype=numpy.int)
            # precision_along_dim = numpy.zeros(self.max_shape[dim], dtype=float)
            # recall_along_dim = numpy.zeros(self.max_shape[dim], dtype=float)
            fme_per_p = numpy.zeros((self.max_shape[dim], 3), dtype=float)
            for p, fme in fme_per_p_dict.items():
                fme_per_p[p] += numpy.asarray(fme).sum(axis=0)
                # counts_along_dim[p] += 1
                # precision_along_dim[p] = self.get_precision(fme)
                # recall_along_dim[p] = self.get_recall(fme)

            # precision_along_dim /= counts_along_dim
            # recall_along_dim /= counts_along_dim
            #
            # precision_along_dim[counts_along_dim == 0] = numpy.nan
            # recall_along_dim[counts_along_dim == 0] = numpy.nan
            # ret[f"bead_precision_along_{self.dim_names[k]}"] = precision_along_dim.tolist()
            # ret[f"bead_recall_along_{self.dim_names[k]}"] = recall_along_dim.tolist()
            # pprint(fme_per_p[numpy.logical_or(fme_per_p[:, 0] != 0, fme_per_p[:, 2] != 0)])
            ret[f"bead_precision_along_{self.dim_names[dim]}"] = [
                1.0 if e == 0 else float(f / (f + e)) for f, m, e in fme_per_p
            ]
            ret[f"bead_recall_along_{self.dim_names[dim]}"] = [
                1.0 if m == 0 else float(f / (f + m)) for f, m, e in fme_per_p
            ]

        return ret
