from typing import List, Optional

import matplotlib.pyplot as plt
import numpy

from matplotlib.figure import Figure
from tifffile import imread

from lnet.utils.plotting import turbo_colormap


def plot_img_projections(img: numpy.ndarray, b=0) -> Figure:
    assert len(img.shape) == 5  # bczyx
    img = img[b]
    assert img.shape[0] == 1  # singleton channel axis
    img = img[0]
    z_len, y_len, x_len = img.shape

    fig = plt.figure(constrained_layout=False)
    gs = fig.add_gridspec(2, 2, width_ratios=[x_len, z_len], height_ratios=[z_len, y_len])
    # ax1
    ax1 = fig.add_subplot(gs[1, 0], anchor="E")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")

    # ax2
    ax2 = fig.add_subplot(gs[1, 1], sharey=ax1, anchor="W")
    # ax2.set_ylabel('y')
    ax2.set_xlabel("z")
    plt.setp(ax2.get_yticklabels(), visible=False)

    # ax3
    ax3 = fig.add_subplot(gs[0, 0], sharex=ax1, anchor="E")
    ax3.set_xlabel("x")
    # ax3.set_ylabel('z')
    plt.setp(ax3.get_xticklabels(), visible=False)

    ax1.imshow(img.max(0), cmap=turbo_colormap)
    ax2.imshow(img.max(2).transpose(), cmap=turbo_colormap)
    ax3.imshow(img.max(1), cmap=turbo_colormap)

    plt.tight_layout()
    fig.subplots_adjust(wspace=0, hspace=0)
    return fig


def plot_img_projections_with_beads(
    img: numpy.ndarray,
    bead_pos: List[numpy.ndarray],
    other_bead_pos: Optional[List[numpy.ndarray]] = None,
    b=0,
    z_min=None,
    z_max=None,
) -> Figure:
    assert len(img.shape) == 5  # bczyx
    img = img[b]
    assert img.shape[0] == 1  # singleton channel axis
    img = img[0]
    z_len, y_len, x_len = img.shape

    bead_pos = bead_pos[b]
    assert len(bead_pos.shape) == 2
    if z_min is not None:
        bead_pos = bead_pos[z_min <= bead_pos[:, 0]]

    if z_max is not None:
        bead_pos = bead_pos[z_max >= bead_pos[:, 0]]

    fig = plt.figure(constrained_layout=False)
    gs = fig.add_gridspec(2, 2, width_ratios=[x_len, z_len], height_ratios=[z_len, y_len])
    # ax1
    ax1 = fig.add_subplot(gs[1, 0], anchor="E")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")

    # ax2
    ax2 = fig.add_subplot(gs[1, 1], sharey=ax1, anchor="W")
    #     ax2.set_ylabel('y')
    ax2.set_xlabel("z")
    plt.setp(ax2.get_yticklabels(), visible=False)

    # ax3
    ax3 = fig.add_subplot(gs[0, 0], sharex=ax1, anchor="E")
    # ax3.set_xlabel('x')
    ax3.set_ylabel("z")
    plt.setp(ax3.get_xticklabels(), visible=False)

    ax1.imshow(img.max(0), cmap=turbo_colormap)
    ax2.imshow(img.max(2).transpose(), cmap=turbo_colormap)
    ax3.imshow(img.max(1), cmap=turbo_colormap)

    ax1.scatter(bead_pos[:, 2], bead_pos[:, 1], c=abs(bead_pos[:, 0] - 25), marker="1")
    ax2.scatter(bead_pos[:, 0], bead_pos[:, 1], c=abs(bead_pos[:, 0] - 25), marker="1")
    ax3.scatter(bead_pos[:, 2], bead_pos[:, 0], c=abs(bead_pos[:, 0] - 25), marker="1")
    if other_bead_pos is not None:
        other_bead_pos = other_bead_pos[b]
        assert len(other_bead_pos.shape) == 2
        ax1.scatter(other_bead_pos[:, 2], other_bead_pos[:, 1], c=abs(other_bead_pos[:, 0] - 25), marker="2")
        ax2.scatter(other_bead_pos[:, 0], other_bead_pos[:, 1], c=abs(other_bead_pos[:, 0] - 25), marker="2")
        ax3.scatter(other_bead_pos[:, 2], other_bead_pos[:, 0], c=abs(other_bead_pos[:, 0] - 25), marker="2")

    plt.tight_layout()
    fig.subplots_adjust(wspace=0, hspace=0)
    return fig


if __name__ == "__main__":
    tgt = (
        imread("K:/beuttenm/repos/lnet/logs/beads/19-08-23_18-32_c307a5a_aux1_/result/test/target/0000.tif")[
            None, ...
        ]
        / numpy.iinfo(numpy.uint16).max
    )
    pred = (
        imread("K:/beuttenm/repos/lnet/logs/beads/19-08-23_18-32_c307a5a_aux1_/result/test/prediction/0000.tif")[
            None, ...
        ]
        / numpy.iinfo(numpy.uint16).max
    )
    plot_img_projections(tgt)
    plt.show()
    plot_img_projections(pred)
    plt.show()
