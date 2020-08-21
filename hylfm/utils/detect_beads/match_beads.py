import logging
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy
from scipy.optimize import linear_sum_assignment
from tifffile import imread

from lnet.utils.detect_beads import get_bead_pos
from lnet.utils.detect_beads.plot_matched_beads import plot_matched_beads

logger = logging.getLogger(__name__)


def match_beads_from_pos(
    btgt: List[numpy.ndarray], bpred: List[numpy.ndarray], *, dist_threshold: float, scaling: Tuple[float, float, float]
) -> Tuple[List[numpy.ndarray], List[numpy.ndarray], List[Tuple[int, int, int]]]:
    assert all(len(tgt.shape) == 2 for tgt in btgt), list(len(tgt.shape) for tgt in btgt)  # bn3
    assert all(len(pred.shape) == 2 for pred in bpred), list(len(pred.shape) for pred in bpred)  # bn3

    assert all(tgt.shape[1] == 3 for tgt in btgt), [tgt.shape for tgt in btgt]  # zyx spatial dims
    assert all(pred.shape[1] == 3 for pred in bpred), [pred.shape for pred in bpred]  # zyx spatial dims

    assert len(btgt) == len(bpred)  # batch dim must match

    btgt_idx = []
    bpred_idx = []
    found_missing_extra = []

    # loop over batch dim
    for tgt, pred in zip(btgt, bpred):
        logger.debug("tgt: %s", tgt.shape)
        logger.debug("pred: %s", pred.shape)
        logger.debug("dist threshold: %s", dist_threshold)

        assert tgt.shape[1] == len(scaling), (tgt.shape, scaling)
        diff = numpy.stack(
            [numpy.subtract.outer(tgt[:, d], pred[:, d]) * s for d, s in zip(range(tgt.shape[1]), scaling)]
        )
        dist = numpy.sqrt(numpy.square(diff).sum(axis=0))
        ridiculous_dist = 1e5
        dist[dist > dist_threshold] = ridiculous_dist
        tgt_idx, pred_idx = linear_sum_assignment(dist)
        logger.debug("solved: %s, %s", tgt_idx.shape, pred_idx.shape)
        valid_dist = dist[tgt_idx, pred_idx]
        valid_dist_mask = valid_dist < ridiculous_dist
        valid_dist = valid_dist[valid_dist_mask]
        nfound = valid_dist.shape[0]
        logger.debug("valid matches: %s", nfound)
        if nfound:
            logger.debug("max dist: %s", numpy.max(valid_dist))
            logger.debug("avg dist: %s", numpy.mean(valid_dist))

        # plt.figure()
        # plt.hist(valid_dist, bins=dist_threshold, range=(0, dist_threshold))
        # plt.show()

        found_missing_extra.append((nfound, tgt.shape[0] - nfound, pred.shape[0] - nfound))
        btgt_idx.append(tgt_idx[valid_dist_mask])
        bpred_idx.append(pred_idx[valid_dist_mask])

    return btgt_idx, bpred_idx, found_missing_extra


def match_beads(
    tgt: numpy.ndarray,
    pred: numpy.ndarray,
    *,
    dist_threshold: float,
    scaling: Tuple[float, float, float],
    min_sigma: float,
    max_sigma: float,
    threshold: float,
    tgt_threshold: float,
    **kwargs
) -> Tuple[
    List[numpy.ndarray], List[numpy.ndarray], List[Tuple[int, int, int]], List[numpy.ndarray], List[numpy.ndarray]
]:
    min_sigma = [min_sigma / s for s in scaling]
    max_sigma = [max_sigma / s for s in scaling]
    kwargs.update(min_sigma=min_sigma, max_sigma=max_sigma)

    bead_pos_pred = get_bead_pos(pred, threshold=threshold, **kwargs)
    # no_beads_found = all([bpp.shape[0] == 0 for bpp in bead_pos_pred])
    # if no_beads_found:
    #     b = len(bead_pos_pred)
    #     return (
    #         [numpy.array([])] * b,
    #         [numpy.array([])] * b,
    #         [(0, -1, 0)] * b,
    #         [numpy.array([])] * b,
    #         [numpy.array([])] * b,
    #     )
    # else:

    bead_pos_tgt = get_bead_pos(tgt, threshold=tgt_threshold, **kwargs)
    btgt_idx, bpred_idx, found_missing_extra = match_beads_from_pos(
        btgt=bead_pos_tgt, bpred=bead_pos_pred, dist_threshold=dist_threshold, scaling=scaling
    )
    return btgt_idx, bpred_idx, found_missing_extra, bead_pos_tgt, bead_pos_pred


if __name__ == "__main__":
    tgt = imread("/g/kreshuk/LF_computed/lnet/logs/beads/01highc/20-04-21_11-41-43/test/output/0/ls.tif")[None, ...]
    print(tgt.max(), tgt.shape)
    pred = imread("/g/kreshuk/LF_computed/lnet/logs/beads/01highc/20-04-21_11-41-43/test/output/0/pred.tif")[None, ...]
    print(pred.max(), pred.shape)

    match_beads_kwargs = {
        "min_sigma": 1.0,
        "max_sigma": 6.0,
        "sigma_ratio": 3.0,
        "threshold": 0.05,
        "overlap": 0.5,
        "exclude_border": False,
        "dist_threshold": 3.0,
        "scaling": [2.0, 1.4, 1.4],
    }

    btgt_idx, bpred_idx, fme, bead_pos_btgt, bead_pos_bpred = match_beads(tgt, pred, **match_beads_kwargs)
    # bead_pos_btgt = get_bead_pos(tgt)
    # bead_pos_bpred = get_bead_pos(pred)
    # btgt_idx, bpred_idx, found_missing_extra = match_beads_from_pos(bead_pos_tgt, bead_pos_pred, dist_threshold=5.0, scaling=(2.0, 1.0, 1.0))
    #
    plot_matched_beads(bead_pos_btgt, btgt_idx, bead_pos_bpred, bpred_idx)
    plt.show()
