from typing import List

import matplotlib.pyplot as plt
import numpy
from matplotlib.figure import Figure


def plot_matched_beads(
    bead_pos_tgt: numpy.ndarray,
    tgt_idx: List[numpy.ndarray],
    bead_pos_pred: numpy.ndarray,
    pred_idx: List[numpy.ndarray],
    bi=0,
) -> Figure:
    assigned = numpy.stack([bead_pos_tgt[bi][tgt_idx[bi]], bead_pos_pred[bi][pred_idx[bi]]])

    fig = plt.figure()
    ax = fig.gca(projection="3d")

    # start at predicted bead pos
    z = assigned[1, :, 0]
    y = assigned[1, :, 1]
    x = assigned[1, :, 2]

    # point at target bead pos
    w = assigned[0, :, 0] - assigned[1, :, 0]
    v = assigned[0, :, 1] - assigned[1, :, 1]
    u = assigned[0, :, 2] - assigned[1, :, 2]

    ax.quiver(x, y, z, u, v, w, length=1.0, normalize=False, arrow_length_ratio=0.5)

    return fig
