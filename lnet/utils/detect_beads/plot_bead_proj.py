import matplotlib.pyplot as plt
import numpy

from tifffile import imread

from lnet.utils.detect_beads import get_bead_pos


def plot_bead_projections(
    bead_pos, other_bead_pos=None, bi=0, z_min=None, z_max=None, color_at=None, other_color_at=None
):
    assert len(bead_pos.shape) == 3
    bead_pos = bead_pos[bi]

    if z_min is not None:
        bead_pos = bead_pos[z_min <= bead_pos[:, 0]]

    if z_max is not None:
        bead_pos = bead_pos[z_max >= bead_pos[:, 0]]

    z_max = bead_pos[:, 0].max()
    y_max = bead_pos[:, 1].max()
    x_max = bead_pos[:, 2].max()

    fig = plt.figure(constrained_layout=False)
    gs = fig.add_gridspec(2, 2, width_ratios=[x_max, z_max], height_ratios=[z_max, y_max])
    # ax1
    ax1 = fig.add_subplot(gs[1, 0], anchor="E")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    # ax2
    ax2 = fig.add_subplot(gs[1, 1], sharey=ax1, anchor="W")
    ax2.set_xlabel("z")
    # ax2.set_ylabel('y')
    plt.setp(ax2.get_yticklabels(), visible=False)
    # ax3
    ax3 = fig.add_subplot(gs[0, 0], sharex=ax1, anchor="E")
    # ax3.set_xlabel('x')
    ax3.set_ylabel("z")
    plt.setp(ax3.get_xticklabels(), visible=False)

    if color_at is None:
        colors = abs(bead_pos[:, 0] - 25)
    else:
        assert len(color_at.shape) == 5
        colors = [
            color_at[b, 0, int(bead_pos[i, 0]), int(bead_pos[i, 1]), int(bead_pos[i, 2])]
            for i in range(bead_pos.shape[0])
        ]

    ax1.scatter(bead_pos[:, 2], bead_pos[:, 1], c=colors, marker="1")
    ax2.scatter(bead_pos[:, 0], bead_pos[:, 1], c=colors, marker="1")
    ax3.scatter(bead_pos[:, 2], bead_pos[:, 0], c=colors, marker="1")
    if other_bead_pos is not None:
        assert len(other_bead_pos.shape) == 3
        other_bead_pos = other_bead_pos[b]
        if other_color_at is None:
            colors = abs(other_bead_pos[:, 0] - 25)
        else:
            assert len(other_color_at.shape) == 5
            colors = [
                other_color_at[b, 0, int(other_bead_pos[i, 0]), int(other_bead_pos[i, 1]), int(other_bead_pos[i, 2])]
                for i in range(other_bead_pos.shape[0])
            ]

        ax1.scatter(other_bead_pos[:, 2], other_bead_pos[:, 1], c=colors, marker="2")
        ax2.scatter(other_bead_pos[:, 0], other_bead_pos[:, 1], c=colors, marker="2")
        ax3.scatter(other_bead_pos[:, 2], other_bead_pos[:, 0], c=colors, marker="2")

    plt.tight_layout()
    fig.subplots_adjust(wspace=0, hspace=0)
    plt.show()


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
    bead_pos_tgt = get_bead_pos(tgt)
    bead_pos_pred = get_bead_pos(pred)
    plot_bead_projections(
        bead_pos_tgt, bead_pos_pred, z_min=10, z_max=40, color_at=tgt.clip(max=0.3), other_color_at=pred.clip(max=0.3)
    )
    plt.show()
