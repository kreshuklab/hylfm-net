import logging
from typing import Iterable, List, Optional, OrderedDict, Sequence

import matplotlib.pyplot as plt
import numpy
from matplotlib import patches
from matplotlib.backends.backend_agg import FigureCanvas
from mpl_toolkits.axes_grid1 import make_axes_locatable

from hylfm.utils.turbo_colormap import turbo_colormap

logger = logging.getLogger(__name__)


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


def get_batch_figure(
    *, tensors: OrderedDict[str, numpy.ndarray], return_array: bool = False, meta: Optional[List[dict]] = None
):
    ncols = len(tensors)
    nrows = tensors[list(tensors.keys())[0]].shape[0]

    fig, ax = plt.subplots(ncols=ncols, nrows=nrows, squeeze=False, figsize=(ncols * 3, nrows * 3))
    if return_array:
        canvas = FigureCanvas(fig)
    else:
        canvas = None

    def make_subplot(ax, title: str, img, boxes: Iterable[Box] = tuple(), side_view=None, with_colorbar=True):
        assert len(img.shape) == 2, img.shape
        if title:
            ax.set_title(title)

        if side_view is not None:
            assert len(side_view.shape) == 2, img.shape
            img = numpy.concatenate(
                [
                    img,
                    numpy.full(shape=(img.shape[0], 5), fill_value=max(img.max(), side_view.max())),
                    numpy.repeat(side_view, max(1, (img.shape[1] / side_view.shape[1]) // 3), axis=1),
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

    for c, (name, tensor) in enumerate(tensors.items()):
        assert tensor.shape[0] == nrows, name
        for r, t in enumerate(tensor):
            t = numpy.squeeze(t)
            if len(t.shape) == 2:
                img = t
                side_view = None
            elif len(t.shape) == 3:
                img = t.max(axis=0)
                side_view = t.max(axis=2).T
            else:
                raise NotImplementedError(t.shape)
                # side_view = vl.max(axis=0).max(axis=2).T

            title = name
            if "slice" in name:
                try:
                    z_slice = meta[r][name]["z_slice"]
                except Exception as e:
                    logger.debug("Could not retrieve z_slice for %s: %s", name, e)
                    try:
                        z_slice = meta[r]["ls_slice"]["z_slice"]
                    except Exception as e:
                        logger.error("Could not retrieve z_slice for fallback ls_slice: %s", e)
                    else:
                        title = f"{name} z: {z_slice}"
                else:
                    title = f"{name} z: {z_slice}"

            make_subplot(ax[r, c], title=title, img=img, side_view=side_view)

    fig.subplots_adjust(hspace=0, wspace=0, bottom=0, top=1, left=0, right=1)
    fig.tight_layout()
    if return_array:
        # Force a draw so we can grab the pixel buffer
        canvas.draw()
        # grab the pixel buffer and dump it into a numpy array
        fig_array = numpy.array(canvas.renderer.buffer_rgba())

        return fig_array
    else:
        return fig
