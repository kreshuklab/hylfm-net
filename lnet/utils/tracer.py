from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import matplotlib.pyplot as plt
import numpy
import scipy.signal
import yaml
from scipy.ndimage import gaussian_filter
from scipy.stats import pearsonr, spearmanr
from skimage.feature import peak_local_max
from torch.utils.data import DataLoader

from lnet import settings
from lnet.datasets import TensorInfo
from lnet.datasets.base import DatasetFromInfo, TiffDataset, get_collate_fn


def trace(
    tgt_path: Path,
    tgt: str,
    plots: List[Union[Dict[str, Dict[str, Union[Path, List]]], Set[str]]],
    output_path: Path,
    overwrite_existing_files: bool = False,
):
    output_path.mkdir(exist_ok=True, parents=True)
    default_smooth = [(("flat", 3), None)]
    all_recons = {}
    for i in range(len(plots)):
        if isinstance(plots[i], set):
            plots[i] = {recon: {"path": tgt_path, "smooth": default_smooth} for recon in plots[i]}
        elif isinstance(plots[i], dict):
            full_recons = {}
            for recon, kwargs in plots[i].items():
                if "path" not in kwargs:
                    kwargs["path"] = tgt_path

                if "smooth" not in kwargs:
                    kwargs["smooth"] = default_smooth

                full_recons[recon] = kwargs

            plots[i] = full_recons
        else:
            raise TypeError(type(plots[i]))

        all_recons.update(plots[i])

    ds_tgt = TiffDataset(info=TensorInfo(name=tgt, root=tgt_path, location=f"{tgt}/*.tif"))
    length = len(ds_tgt)
    datasets_to_trace = {
        recon: TiffDataset(info=TensorInfo(name=recon, root=kwargs["path"], location=f"{recon}/*.tif"))
        for recon, kwargs in all_recons.items()
    }
    assert all([len(recon_ds) == length for recon_ds in datasets_to_trace.values()]), [length] + [
        len(recon_ds) for recon_ds in datasets_to_trace.values()
    ]
    assert tgt not in datasets_to_trace
    datasets_to_trace[tgt] = ds_tgt

    figs = {}
    peak_path = tgt_path / f"{tgt}_peaks.yml"
    nr_traces = 1
    peaks = None
    if peak_path.exists() and not overwrite_existing_files:
        with peak_path.open() as f:
            peaks = numpy.asarray(yaml.safe_load(f))

        if peaks.shape[0] != nr_traces:
            peaks = None

    if peaks is None:
        tgt_min_path = tgt_path / f"{tgt}_min.npy"
        tgt_max_path = tgt_path / f"{tgt}_max.npy"
        if tgt_min_path.exists() and tgt_max_path.exists() and not overwrite_existing_files:
            min_tensor = numpy.load(str(tgt_min_path))
            max_tensor = numpy.load(str(tgt_max_path))
        else:
            min_tensor, max_tensor = get_min_max(ds_tgt, tgt)
            numpy.save(str(tgt_min_path), min_tensor)
            numpy.save(str(tgt_max_path), max_tensor)

        diff_tensor = max_tensor - min_tensor
        smooth_diff_tensor = gaussian_filter(diff_tensor, sigma=1.3, mode="constant")
        # diff_tensor = gaussian_filter(max_tensor, sigma=1.3, mode="constant") - gaussian_filter(min_tensor, sigma=1.3, mode="constant")
        # blobs = blob_dog(
        #     diff_tensor, min_sigma=1, max_sigma=16, sigma_ratio=1.6, threshold=.3, overlap=0.5, exclude_border=True
        # )
        # peaks = blob_dog(
        #     diff_tensor, min_sigma=1.0, max_sigma=5, sigma_ratio=1.1, threshold=.1, overlap=0.5, exclude_border=False
        # )
        # smooth_diff_tensor = diff_tensor
        # smooth_diff_tensor = gaussian_filter(diff_tensor, sigma=1.3, mode="constant")
        peaks = peak_local_max(
            smooth_diff_tensor, min_distance=3, threshold_abs=1.0, exclude_border=True, num_peaks=nr_traces
        )
        r = 6  # same radius for all
        peaks = numpy.concatenate([peaks, numpy.full((peaks.shape[0], 1), r)], axis=1)
        peaks_on = {"diff tensor": diff_tensor, "smooth diff tensor": smooth_diff_tensor}
        for name, tensor in peaks_on.items():
            # plot peak positions on smoothed diff tensor
            fig, ax = plt.subplots()
            title = f"peaks on {name}"
            ax.set_title(title)
            im = ax.imshow(tensor.squeeze())
            fig.colorbar(im, ax=ax)
            for i, peak in enumerate(peaks):
                y, x, r = peak
                c = plt.Circle((x, y), r, color="r", linewidth=1, fill=False)
                plt.text(x + 2 * int(r + 0.5), y, str(i))
                ax.add_patch(c)

            ax.set_axis_off()
            plt.savefig(tgt_path / f"{tgt}_{title.replace(' ', '_')}.svg")
            # plt.show()
            figs[name.replace(" ", "_")] = fig

        with peak_path.open("w") as f:
            yaml.safe_dump(peaks.tolist(), f)

    all_traces = {}
    for el, p in {tgt: tgt_path, **{recon: kwargs["path"] for recon, kwargs in all_recons.items()}}.items():
        traces_path: Path = p / f"{el}_traces.npy"
        if traces_path.exists() and not overwrite_existing_files:
            all_traces[el] = numpy.load(str(traces_path))
        else:
            all_traces[el] = trace_peaks(datasets_to_trace[el], el, peaks)
            numpy.save(str(traces_path), all_traces[el])

    all_smooth_traces = {}
    correlations = {}
    for recon, kwargs in all_recons.items():
        for smooth, tgt_smooth in kwargs["smooth"]:
            traces, tgt_traces = get_smooth_traces_pair(recon, smooth, tgt, tgt_smooth, all_traces, all_smooth_traces)
            if (recon, smooth, tgt_smooth, 0) not in correlations:
                for t, (trace, tgt_trace) in enumerate(zip(traces, tgt_traces)):
                    pr, _ = pearsonr(trace, tgt_trace)
                    sr, _ = spearmanr(trace, tgt_trace)
                    correlations[recon, smooth, tgt_smooth, t] = {"pearson": pr, "spearman": sr}

    trace_figs = plot_traces(
        tgt=tgt,
        plots=plots,
        all_traces=all_traces,
        all_smooth_traces=all_smooth_traces,
        correlations=correlations,
        output_path=output_path,
    )
    figs.update(trace_figs)
    return peaks, all_traces, correlations, figs


def get_min_max(ds: DatasetFromInfo, name: str):
    min_ = max_ = None
    for sample in DataLoader(
        dataset=ds,
        shuffle=False,
        collate_fn=get_collate_fn(lambda b: b),
        num_workers=settings.max_workers_for_trace,
        pin_memory=False,
    ):
        tensor = sample[name].squeeze()
        if min_ is None:
            min_ = tensor
            max_ = tensor
        else:
            min_ = numpy.minimum(min_, tensor)
            max_ = numpy.maximum(max_, tensor)

    assert len(min_.shape) == 2, min_.shape
    assert len(max_.shape) == 2, max_.shape
    return min_, max_


def trace_peaks(ds: DatasetFromInfo, name: str, peaks: numpy.ndarray, reduce: str = "mean"):
    assert len(peaks.shape) == 2, peaks.shape
    assert peaks.shape[1] == 3, peaks.shape

    if reduce == "mean":
        pass
    else:
        raise NotImplementedError(reduce)

    h, w = ds[0][name].squeeze().shape
    peak_masks = numpy.stack([create_circular_mask(h, w, p).flatten() for p in peaks], axis=1)
    cirlce_area = peak_masks[:, 0].sum()
    assert len(peak_masks.shape) == 2, peak_masks.shape
    assert peak_masks.shape[0] == h * w, (peak_masks.shape, h, w)
    assert peak_masks.shape[1] == peaks.shape[0], (peak_masks.shape, peaks.shape)

    traces = numpy.empty(shape=(peaks.shape[0], len(ds)))
    for i, sample in enumerate(
        DataLoader(
            dataset=ds,
            shuffle=False,
            collate_fn=get_collate_fn(lambda b: b),
            num_workers=settings.max_workers_for_trace,
            pin_memory=False,
        )
    ):
        tensor = sample[name].flatten()
        traces[:, i] = tensor.dot(peak_masks)

    traces /= cirlce_area
    return traces


def create_circular_mask(h: int, w: int, peak: Tuple[int, int, int]):
    center = peak[:2]
    radius = peak[2]
    H, W = numpy.ogrid[:h, :w]
    dist_from_center = numpy.sqrt((H - center[0]) ** 2 + (W - center[1]) ** 2)

    mask = dist_from_center <= radius
    return mask


def get_smooth_traces(traces, wname, w):
    window = scipy.signal.get_window(wname, w, fftbins=False)
    window /= window.sum()
    return numpy.stack([numpy.convolve(window, trace, mode="valid") for trace in traces])


def get_smooth_traces_pair(
    recon: str,
    smooth: Optional[Tuple[str, int]],
    tgt: str,
    tgt_smooth: Optional[Tuple[str, int]],
    all_traces: dict,
    all_smooth_traces: dict,
):
    if smooth is None:
        traces = all_traces[recon]
        w = 0
    else:
        wname, w = smooth
        smooth_traces = all_smooth_traces.get((recon, wname, w), None)
        if smooth_traces is None:
            smooth_traces = get_smooth_traces(all_traces[recon], wname, w)
            all_smooth_traces[recon, wname, w] = smooth_traces

        traces = smooth_traces

    if tgt_smooth is None:
        tgt_traces = all_traces[tgt]
        tgt_w = 0
    else:
        tgt_wname, tgt_w = smooth
        smooth_tgt_traces = all_smooth_traces.get((tgt, tgt_wname, tgt_w), None)
        if smooth_tgt_traces is None:
            smooth_tgt_traces = get_smooth_traces(all_traces[tgt], tgt_wname, tgt_w)
            all_smooth_traces[tgt, tgt_wname, tgt_w] = smooth_tgt_traces

        tgt_traces = smooth_tgt_traces

    if w < tgt_w:
        wdiff = tgt_w - w
        traces = traces[:, wdiff // 2 : -(wdiff // 2)]
    elif tgt_w < w:
        wdiff = w - tgt_w
        tgt_traces = tgt_traces[:, wdiff // 2 : -(wdiff // 2)]

    return traces, tgt_traces


def plot_traces(
    *, tgt, plots, all_traces, all_smooth_traces, correlations, output_path,
):
    trace_name_map = {"pred": "LFN", "ls_slice": "LS", "lr_slice": "LR"}
    tgt_cmap = plt.get_cmap("Blues").reversed()

    sequential_cmaps = [
        plt.get_cmap(name).reversed()
        for name in [
            "Greens",
            "Oranges",
            "Reds",
            "Greys",
            "Purples",
            "YlOrBr",
            "YlOrRd",
            "OrRd",
            "PuRd",
            "RdPu",
            "BuPu",
            "GnBu",
            "PuBu",
            "YlGnBu",
            "PuBuGn",
            "BuGn",
            "YlGn",
        ]
    ]  # 'Blues'

    nr_color_shades = max([len(kwargs["smooth"]) for recons in plots for kwargs in recons.values()])

    def get_color(i: Optional[int], j):
        if i is None:
            return tgt_cmap(j / nr_color_shades * 0.75)
        else:
            return sequential_cmaps[i](j / nr_color_shades * 0.75)

    plot_kwargs = {"linewidth": 1}

    def plot_centered(*xy, ax, **kwargs):
        y: numpy.ndarray = xy[-1]
        y_min = numpy.min(y)
        y_max = numpy.max(y)
        y_center = numpy.median(y)
        half_range = max(abs(y_center - y_min), abs(y_center - y_max))
        ax.set_ylim(y_center - half_range, y_center + half_range)
        return ax.plot(*xy, **kwargs)

    def plot_trace(ax, j, traces, t, name, *, i=None, wname="", w=0, w_max=0):
        label = f"{trace_name_map.get(name, name):>3} {wname:>5} {w:<1} "
        if (name, wname, w, t) in correlations:
            label += (
                f"Pearson: {correlations[name, wname, w, t]['pearson'][t]:.3f}  "
                f"Spearman: {correlations[name, wname, w, t]['spearman'][t]:.3f}"
            )
        return plot_centered(
            numpy.arange(w_max // 2, all_traces[tgt].shape[1] - w_max // 2),
            traces[t],
            ax=ax,
            label=label,
            color=get_color(i, j),
            **plot_kwargs,
        )

    # from https://matplotlib.org/3.1.1/gallery/ticks_and_spines/multiple_yaxis_with_spines.html
    def make_patch_spines_invisible(ax):
        ax.set_frame_on(True)
        ax.patch.set_visible(False)
        for sp in ax.spines.values():
            sp.set_visible(False)

    fontdict = {"family": "monospace"}
    figs = {}
    rel_dist_per_recon = 0.03
    for t in range(all_traces[tgt].shape[0]):
        nrows = len(plots)
        fig, axes = plt.subplots(nrows=nrows, sharex=True, figsize=(20, 4), squeeze=False)
        axes = axes[:, 0]  # only squeeze ncols=1
        # plt.suptitle(f"Trace {i:2}")
        axes[0].set_title(f"Trace {t:2}")
        for ax in axes:
            ax.tick_params(axis="y", labelcolor=get_color(None, 0))
            ax.set_xlim(0, all_traces[tgt].shape[1])
            ax.set_ylabel(trace_name_map.get(tgt, tgt), color=get_color(None, 0), fontdict=fontdict)

        i = 0
        j = 0
        for ax, recons in zip(axes, plots):
            plotted_lines = []
            for i, (recon, kwargs) in enumerate(recons.items()):
                twinx = ax.twinx()  # new twinx for each recon
                twinx.tick_params(axis="y", labelcolor=get_color(i, 0))
                if i:
                    # offset second and higher twinx
                    # adapted from https://matplotlib.org/3.1.1/gallery/ticks_and_spines/multiple_yaxis_with_spines.html
                    twinx.spines["right"].set_position(("axes", 1.0 + rel_dist_per_recon * i))
                    # Having been created by twinx(), twinx has its frame off, so the line of its detached spine is
                    # invisible.  First, activate the frame but make the patch and spines invisible.
                    make_patch_spines_invisible(twinx)
                    # Second, show the right spine.
                    twinx.spines["right"].set_visible(True)

                for smooth, tgt_smooth in kwargs["smooth"]:
                    traces, tgt_traces = get_smooth_traces_pair(
                        recon, smooth, tgt, tgt_smooth, all_traces, all_smooth_traces
                    )
                    if tgt_smooth is None:
                        tgt_wname = ""
                        tgt_w = 0
                    else:
                        tgt_wname, tgt_w = tgt_smooth

                    if smooth is None:
                        wname = ""
                        w = 0
                    else:
                        wname, w = smooth

                    w_max = max(w, tgt_w)
                    plotted_lines += plot_trace(ax, j, tgt_traces, t, tgt, wname=tgt_wname, w=w, w_max=w_max)
                    plotted_lines += plot_trace(twinx, j, traces, t, tgt, wname=wname, w=w, w_max=w_max)
                    j += 1

            labels = [l.get_label() for l in plotted_lines]
            ax.legend(
                plotted_lines,
                labels,
                bbox_to_anchor=(1.0 + rel_dist_per_recon * (i + 1), 0.5),
                loc="center left",
                prop=fontdict,
            )
            # plotted_lines[0] += plot_trace(
            #     twinx_axes[0], 0, traces[name][t], name, i=i
            # )  # plot unsmoothed recon to first subfig
            # for j, (wname, w) in enumerate(smooth_with, start=1):
            #     axj = j if individual_subfigures else 1
            #     # twinx_axes[axj].set_ylabel(trace_name_map.get(name, name), fontdict=fontdict)
            #     plotted_lines[axj] += plot_trace(
            #         twinx_axes[axj], j, all_smooth_traces[wname, w][t], name, i=i, wname=wname, w=w
            #     )

        figs[f"trace{t}"] = fig
        plt.savefig(output_path / f"trace{t}.svg")

    return figs


if __name__ == "__main__":
    peaks, traces, correlations, figs = trace(
        # tgt_path = Path("/g/kreshuk/LF_computed/lnet/logs/brain1/z_out49/f2_only11_2/20-05-19_12-27-16/test/run000/ds0-0")
        tgt_path=Path(
            "/g/kreshuk/LF_computed/lnet/logs/brain1/test_z_out49/lr_f4/20-05-31_21-24-03/brain.09_3__2020-03-09_06.43.40__SinglePlane_-330/run000/ds0-0"
        ),
        tgt="ls_slice",
        plots=[{"lr_slice"}],
        output_path=Path("/g/kreshuk/LF_computed/lnet/traces"),
    )
    # peaks, traces, correlations, figs = trace(
    #     tgt_path=Path(
    #         "/g/kreshuk/LF_computed/lnet/logs/brain1/test_z_out49/lr_f4/20-05-31_21-24-03/brain.09_3__2020-03-09_06.43.40__SinglePlane_-330/run000/ds0-0"
    #     ),
    #     tgt="ls_slice",
    #     plots=[
    #         {
    #             "pred": {
    #                 "path": Path(
    #                     "/g/kreshuk/LF_computed/lnet/logs/brain1/test_z_out49/f4/z_out49/f4_b2_only09_3/20-05-30_10-41-55/v1_checkpoint_82000_MSSSIM=0.8523718668864324/brain.09_3__2020-03-09_06.43.40__SinglePlane_-330/run000/ds0-0/"
    #                 )
    #             },
    #             "lr_slice": {
    #                 "path": Path(
    #                     "/g/kreshuk/LF_computed/lnet/logs/brain1/test_z_out49/lr_f4/20-05-31_21-24-03/brain.09_3__2020-03-09_06.43.40__SinglePlane_-330/run000/ds0-0"
    #                 )
    #             },
    #         }
    #     ],
    #     output_path=Path("/g/kreshuk/LF_computed/lnet/traces"),
    # )
    for name, trace in traces.items():
        print(name, trace.shape, trace.min(), trace.max())

    plt.show()
