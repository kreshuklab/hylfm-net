import matplotlib.pyplot as plt
import numpy
from matplotlib.figure import Figure
from tifffile import imread

from lnet.utils.detect_beads import get_bead_pos


def plot_bead_hist(bead_pos_tgt, bead_pos_pred, bi=0) -> Figure:
    z_tgt_counts, bin_edges = numpy.histogram(bead_pos_tgt[bi][:, 0], bins=49, range=(0, 49))
    z_pred_counts, bin_edges = numpy.histogram(bead_pos_pred[bi][:, 0], bins=49, range=(0, 49))

    correct = numpy.minimum(z_pred_counts, z_tgt_counts)
    diff = z_tgt_counts - z_pred_counts
    missing = numpy.clip(diff, a_min=0, a_max=numpy.inf)
    extra = numpy.clip(-diff, a_min=0, a_max=numpy.inf)

    fig, ax = plt.subplots()
    width = 0.35  # the width of the bars
    x = numpy.arange(0.5, 49.5)
    # ax.bar(x - width/2, z_tgt_counts, width, label="tgt")
    # ax.bar(x + width/2, z_pred_counts, width, label="pred")
    # ax.bar(x - width/2, z_pred_counts - z_tgt_counts, width, label="diff abs")
    # ax.bar(x + width/2, (z_pred_counts - z_tgt_counts) / z_tgt_counts, width, label="diff rel")
    ax.bar(x, correct, 1, color="g", label="correct")
    ax.bar(x, missing, 1, bottom=correct, color="r", label="missing")
    ax.bar(x, extra, 1, bottom=correct, color="b", label="extra")

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
    bead_pos_tgt = get_bead_pos(tgt)
    bead_pos_pred = get_bead_pos(pred)
    plot_bead_hist(bead_pos_tgt, bead_pos_pred)
    plt.show()
