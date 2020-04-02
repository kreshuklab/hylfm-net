from typing import Sequence, List, Iterable

import numpy
from matplotlib import patches
from mpl_toolkits.axes_grid1 import make_axes_locatable

from lnet.utils.turbo_colormap import turbo_colormap


class Box:
    def __init__(self, slice_x: slice, slice_y: slice, color: str):
        self.slice_x = slice_x
        self.slice_y = slice_y
        self.color = color

    def apply_to_ax(self, ax):
        box = patches.Rectangle(
            (self.slice_x.start, self.slice_y.start),
            self.slice_x.stop - self.slice_x.start,
            self.slice_y.stop - self.slice_y.start,
            linewidth=1,
            edgecolor=self.color,
            facecolor="none",
        )
        ax.add_patch(box)


class ColorSelection:
    def __init__(self, colors: Sequence[str]):
        self.colors = colors

    def __getitem__(self, item: int):
        return self.colors[item % len(self.colors)]


def get_batch_figure(*, ipt_batch: numpy.ndarray, pred_batch: numpy.ndarray, tgt_batch: numpy.ndarray, voxel_losses: List[numpy.ndarray], boxes: Iterable[Box] = tuple()):
    if len(tgt_batch.shape) == 1:
        tgt_batch = [None] * ipt_batch.shape[0]
    else:
        assert ipt_batch.shape[0] == tgt_batch.shape[0], (ipt_batch.shape, tgt_batch.shape)
        assert len(tgt_batch.shape) in (4, 5), tgt_batch.shape
        assert tgt_batch.shape[1] == 1, tgt_batch.shape

    has_aux = False  # hasattr(output, "aux_tgt")
    if has_aux:
        raise NotImplementedError
        # aux_tgt_batch = numpy.stack([yy.cpu().numpy() for yy in output.aux_tgt])
        # aux_pred_batch = numpy.stack([yy.detach().cpu().numpy() for yy in output.aux_pred])
        # assert ipt_batch.shape[0] == aux_tgt_batch.shape[0], (ipt_batch.shape, aux_tgt_batch.shape)
        # assert len(aux_tgt_batch.shape) == 5, aux_tgt_batch.shape
        # assert aux_tgt_batch.shape[1] == 1, aux_tgt_batch.shape
    else:
        aux_tgt_batch = None
        aux_pred_batch = None

    ncols = 5 + 2 * int(has_aux) + len(voxel_losses)

    nrows = ipt_batch.shape[0]
    fig, ax = plt.subplots(ncols=ncols, nrows=nrows, squeeze=False, figsize=(ncols * 3, nrows * 3))

    def make_subplot(
        ax_list, title: str, img, boxes: Iterable[Box] = tuple(), side_view=None, with_colorbar=True
    ):
        global col
        ax = ax_list[col]
        if title:
            ax.set_title(title)

        if side_view is not None:
            img = numpy.concatenate(
                [
                    img,
                    numpy.full(shape=(img.shape[0], 1), fill_value=side_view.max()),
                    numpy.repeat(side_view, 3, axis=1),
                ],
                axis=1,
            )

        im = ax.imshow(img, cmap=turbo_colormap)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.axis("off")
        for box in boxes:
            box.apply_to_ax(ax)

        if with_colorbar:
            # from https://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
            # create an axes on the right side of ax. The width of cax will be 5%
            # of ax and the padding between cax and ax will be fixed at 0.05 inch.
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="3%", pad=0.03)
            fig.colorbar(im, cax=cax)
            # ax.set_title(f"min-{img.min():.2f}-max-{img.max():.2f}")  # taking too much space!

        col += 1

    global col
    for i, (ib, tb, pb) in enumerate(zip(ipt_batch, tgt_batch, pred_batch)):
        if tb is not None and len(tb.shape) == 4:
            assert tb.shape[0] == 1
            tb = tb[0]

        if len(pb.shape) == 4:
            assert pb.shape[0] == 1
            pb = pb[0]

        col = 0
        make_subplot(ax[i], "", ib[0])
        make_subplot(ax[i], "prediction", pb.max(axis=0), boxes=boxes, side_view=pb.max(axis=2).T)
        if tb is not None:
            make_subplot(ax[i], "target", tb.max(axis=0), boxes=boxes, side_view=tb.max(axis=2).T)
            tb_abs = numpy.abs(tb) + 0.1
            pb_abs = numpy.abs(pb) + 0.1
            rel_diff = numpy.max([tb_abs / pb_abs, pb_abs / tb_abs], axis=0)
            abs_diff = numpy.abs(pb - tb)
            make_subplot(ax[i], "rel diff", rel_diff.max(axis=0), side_view=rel_diff.max(axis=2).T)
            make_subplot(ax[i], "abs_diff", abs_diff.max(axis=0), side_view=abs_diff.max(axis=2).T)

    if has_aux:
        col_so_far = col
        for i, (atb, apb) in enumerate(zip(aux_tgt_batch, aux_pred_batch)):
            if len(atb.shape) == 4:
                assert atb.shape[0] == 1
                atb = atb[0]

            if len(apb.shape) == 4:
                assert apb.shape[0] == 1
                apb = apb[0]

            col = col_so_far
            make_subplot(ax[i], "aux tgt", atb.max(axis=0), boxes=boxes)
            make_subplot(ax[i], "aux pred", apb.max(axis=0), boxes=boxes)

    col_so_far = col
    for loss_nr, vl_batch in enumerate(voxel_losses):
        for i, vl in enumerate(vl_batch):
            if len(vl.shape) ==  3:
                vl = vl[None, ...]
            col = col_so_far
            make_subplot(
                ax[i],
                f"voxel loss {loss_nr}",
                vl.max(axis=0).max(axis=0),
                boxes=boxes,
                side_view=vl.max(axis=0).max(axis=2).T,
            )

    fig.subplots_adjust(hspace=0, wspace=0, bottom=0, top=1, left=0, right=1)
    fig.tight_layout()
    return fig