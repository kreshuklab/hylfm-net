import json
import logging
import math
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import imageio
import matplotlib.pyplot as plt
import numpy
import scipy.signal
import skvideo.io
import skvideo.motion
import yaml
from scipy.ndimage import gaussian_filter
from scipy.stats import pearsonr, spearmanr
from skimage.feature import peak_local_max
from torch.utils.data import DataLoader, Subset

from lnet import settings
from lnet.datasets import TensorInfo
from lnet.datasets.base import TiffDataset, get_collate_fn

logger = logging.getLogger(__name__)


def trace(
    tgt_path: Path,
    tgt: str,
    roi: Tuple[slice, slice],
    plots: List[Union[Dict[str, Dict[str, Union[Path, List]]], Set[str]]],
    output_path: Path,
    nr_traces: int,
    overwrite_existing_files: bool = False,
    smooth_diff_sigma: float = 1.3,
    peak_threshold_abs: float = 1.0,
    reduce_peak_area: str = "mean",
    time_range: Optional[Tuple[int, Optional[int]]] = None,
    plot_peaks: bool = False,
    compute_peaks_on: str = "std",  # std, diff, a*std+b*diff
    peaks_min_dist: int = 3,
    trace_radius: int = 3,
    compensate_motion: Optional[dict] = None,
    tag: str = "",  # for plot title only
):
    output_path /= f"tgt-{tgt}_diffsigma-{smooth_diff_sigma}_pthreshabs-{peak_threshold_abs}_red-{reduce_peak_area}_tr-{time_range}_roi-{roi}"
    output_path.mkdir(exist_ok=True, parents=True)
    default_smooth = [(None, ("flat", 3))]
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

    all_recon_paths = {}
    for recons in plots:
        for recon, kwargs in recons.items():
            if recon in all_recon_paths:
                assert kwargs["path"] == all_recon_paths[recon], (kwargs["path"], all_recon_paths[recon])
            else:
                all_recon_paths[recon] = kwargs["path"]

    ds_tgt = TiffDataset(info=TensorInfo(name=tgt, root=tgt_path, location=f"{tgt}/*.tif"))
    length = len(ds_tgt)
    datasets_to_trace = {
        recon: TiffDataset(info=TensorInfo(name=recon, root=path, location=f"{recon}/*.tif"))
        for recon, path in all_recon_paths.items()
    }
    assert all([len(recon_ds) == length for recon_ds in datasets_to_trace.values()]), [length] + [
        len(recon_ds) for recon_ds in datasets_to_trace.values()
    ]
    assert tgt not in datasets_to_trace
    datasets_to_trace[tgt] = ds_tgt
    if time_range is not None:
        for name, ds in datasets_to_trace.items():
            datasets_to_trace[name] = Subset(
                ds, numpy.arange(time_range[0], len(ds) if time_range[1] is None else time_range[1])
            )

    # load data
    for name, ds in datasets_to_trace.items():
        datasets_to_trace[name] = numpy.stack(
            [
                sample[name].squeeze()[roi]
                for sample in DataLoader(
                    dataset=ds,
                    shuffle=False,
                    collate_fn=get_collate_fn(lambda b: b),
                    num_workers=settings.max_workers_for_trace,
                    pin_memory=False,
                )
            ]
        )
        assert numpy.isfinite(datasets_to_trace[name]).all()

    if compensate_motion is not None:
        compensate_ref_name = compensate_motion.pop("compensate_ref", None)
        compensate_ref = datasets_to_trace[compensate_ref_name]
        motion = skvideo.motion.blockMotion(compensate_ref, **compensate_motion)
        print("motion", motion.shape, motion.max())
        assert numpy.isfinite(motion).all()
        for name in set(datasets_to_trace.keys()):
            data = datasets_to_trace[name]
            print("data", data.shape)

            # compensate the video
            compensate = (
                skvideo.motion.blockComp(data, motion, mbSize=compensate_motion.get("mbSize", 8))
                .squeeze(axis=-1)
                .astype("float32")
            )
            print("compensate", compensate.shape)
            assert numpy.isfinite(compensate).all()
            # write
            imageio.volwrite(output_path / f"{name}_motion_compensated.tif", compensate)
            imageio.volwrite(output_path / f"{name}_not_compensated.tif", data)
    else:
        motion = None

    figs = {}
    peak_path = tgt_path / f"{tgt}_peaks_of_{compute_peaks_on}.yml"
    peaks = None
    if peak_path.exists() and not overwrite_existing_files:
        with peak_path.open() as f:
            peaks = numpy.asarray(yaml.safe_load(f))

        if peaks.shape[0] != nr_traces:
            peaks = None

    if peaks is None or plot_peaks:
        recompute_traces = True
        all_projections = OrderedDict()
        for tensor_name, path in {tgt: tgt_path, **all_recon_paths}.items():
            min_path = path / f"{tensor_name}_min.npy"
            max_path = path / f"{tensor_name}_max.npy"
            mean_path = path / f"{tensor_name}_mean.npy"
            std_path = path / f"{tensor_name}_std.npy"

            if (
                min_path.exists()
                and max_path.exists()
                and mean_path.exists()
                and std_path.exists()
                and not overwrite_existing_files
            ):
                min_tensor = numpy.load(str(min_path))
                max_tensor = numpy.load(str(max_path))
                mean_tensor = numpy.load(str(mean_path))
                std_tensor = numpy.load(str(std_path))
            else:
                min_tensor, max_tensor, mean_tensor, std_tensor = get_min_max_mean_std(datasets_to_trace[tensor_name])
                numpy.save(str(min_path), min_tensor)
                numpy.save(str(max_path), max_tensor)
                numpy.save(str(mean_path), mean_tensor)
                numpy.save(str(std_path), std_tensor)

            all_projections[tensor_name] = {
                "min": min_tensor,
                "max": max_tensor,
                "mean": mean_tensor,
                "std": std_tensor,
            }

        all_projections.move_to_end(tgt, last=False)
        # tgt_min_path = tgt_path / f"{tgt}_min.npy"
        # tgt_max_path = tgt_path / f"{tgt}_max.npy"
        # if tgt_min_path.exists() and tgt_max_path.exists() and not overwrite_existing_files:
        #     min_tensor = numpy.load(str(tgt_min_path))
        #     max_tensor = numpy.load(str(tgt_max_path))
        # else:
        #     min_tensor, max_tensor = get_min_max_std(ds_tgt, tgt)
        #     numpy.save(str(tgt_min_path), min_tensor)
        #     numpy.save(str(tgt_max_path), max_tensor)

        # diff_tensor = gaussian_filter(max_tensor, sigma=1.3, mode="constant") - gaussian_filter(min_tensor, sigma=1.3, mode="constant")
        # blobs = blob_dog(
        #     diff_tensor, min_sigma=1, max_sigma=16, sigma_ratio=1.6, threshold=.3, overlap=0.5, exclude_border=True
        # )
        # peaks = blob_dog(
        #     diff_tensor, min_sigma=1.0, max_sigma=5, sigma_ratio=1.1, threshold=.1, overlap=0.5, exclude_border=False
        # )
        # smooth_diff_tensor = diff_tensor
        # smooth_diff_tensor = gaussian_filter(diff_tensor, sigma=1.3, mode="constant")

        for tensor_name, projections in all_projections.items():
            diff_tensor = projections["max"] - projections["min"]
            plot_peaks_on = {
                "diff tensor": diff_tensor,
                "min tensor": projections["min"],
                "max tensor": projections["max"],
                "std tensor": projections["std"],
                "mean tensor": projections["mean"],
            }

            def get_peaks_on_tensor(_comp_on=compute_peaks_on):
                if _comp_on == "diff":
                    return diff_tensor
                elif _comp_on == "smooth_diff":
                    peaks_on_tensor = gaussian_filter(diff_tensor, sigma=smooth_diff_sigma, mode="constant")
                    plot_peaks_on["smooth diff tensor"] = peaks_on_tensor
                elif _comp_on == "std":
                    peaks_on_tensor = projections["std"]
                elif "+" in _comp_on:
                    comp_on_parts = _comp_on.split("+")
                    peaks_on_tensor = get_peaks_on_tensor(comp_on_parts[0])
                    for part in comp_on_parts[1:]:
                        peaks_on_tensor = numpy.add(peaks_on_tensor, get_peaks_on_tensor(part))
                elif "*" in _comp_on:
                    factor, comp_on = _comp_on.split("*")
                    factor = float(factor)
                    peaks_on_tensor = factor * get_peaks_on_tensor(comp_on)
                else:
                    raise NotImplementedError(compute_peaks_on)

                return peaks_on_tensor

            peaks_on_tensor = get_peaks_on_tensor()
            if tensor_name == tgt:
                peaks = peak_local_max(
                    peaks_on_tensor,
                    min_distance=peaks_min_dist,
                    threshold_abs=peak_threshold_abs,
                    exclude_border=True,
                    num_peaks=nr_traces,
                )
                peaks = numpy.concatenate([peaks, numpy.full((peaks.shape[0], 1), trace_radius)], axis=1)

            # plot peak positions on different projections
            fig, axes = plt.subplots(nrows=math.ceil(len(plot_peaks_on) / 3), ncols=3, squeeze=False, figsize=(20, 10))
            plt.suptitle(tensor_name)
            for ax, (name, tensor) in zip(axes.flatten(), plot_peaks_on.items()):
                title = f"peaks on {name}"
                ax.set_title(title)
                im = ax.imshow(tensor)
                fig.colorbar(im, ax=ax)
                for i, peak in enumerate(peaks):
                    y, x, r = peak
                    c = plt.Circle((x, y), r, color="r", linewidth=1, fill=False)
                    ax.text(x + 2 * int(r + 0.5), y, str(i))  # todo fix text
                    ax.add_patch(c)

                ax.set_axis_off()
                plt.savefig(tgt_path / f"{tgt}_{title.replace(' ', '_')}.svg")
                figs[name.replace(" ", "_")] = fig

            with peak_path.open("w") as f:
                yaml.safe_dump(peaks.tolist(), f)
    else:
        recompute_traces = False

    all_traces = {}
    for el, p in {tgt: tgt_path, **all_recon_paths}.items():
        traces_path: Path = p / f"{el}_traces.npy"
        if traces_path.exists() and not overwrite_existing_files and not recompute_traces:
            all_traces[el] = numpy.load(str(traces_path))
        else:
            all_traces[el] = trace_peaks(datasets_to_trace[el], el, peaks, reduce_peak_area)
            numpy.save(str(traces_path), all_traces[el])

    all_smooth_traces = {}
    correlations = {}
    for recons in plots:
        for recon, kwargs in recons.items():
            for smooth, tgt_smooth in kwargs["smooth"]:
                if (recon, smooth, tgt_smooth, 0) not in correlations:
                    traces, tgt_traces = get_smooth_traces_pair(
                        recon, smooth, tgt, tgt_smooth, all_traces, all_smooth_traces
                    )
                    for t, (trace, tgt_trace) in enumerate(zip(traces, tgt_traces)):
                        try:
                            pr, _ = pearsonr(trace, tgt_trace)
                        except ValueError as e:
                            logger.error(e)
                            pr = numpy.nan

                        try:
                            sr, _ = spearmanr(trace, tgt_trace)
                        except ValueError as e:
                            logger.error(e)
                            sr = numpy.nan

                        correlations[recon, smooth, tgt_smooth, t] = {"pearson": pr, "spearman": sr}

    trace_figs = plot_traces(
        tgt=tgt,
        plots=plots,
        all_traces=all_traces,
        all_smooth_traces=all_smooth_traces,
        correlations=correlations,
        output_path=output_path,
        tag=tag,
    )
    figs.update(trace_figs)
    return peaks, all_traces, correlations, figs, motion


# def get_min_max_mean_std(ds: DatasetFromInfo, name: str):
#     min_ = max_ = None
#     means = []
#     vars = []
#     tensor = numpy.stack()
#     for sample in DataLoader(
#         dataset=ds,
#         shuffle=False,
#         collate_fn=get_collate_fn(lambda b: b),
#         num_workers=settings.max_workers_for_trace,
#         pin_memory=False,
#     ):
#         tensor = sample[name].squeeze()
#         if min_ is None:
#             min_ = tensor
#             max_ = tensor
#         else:
#             min_ = numpy.minimum(min_, tensor)
#             max_ = numpy.maximum(max_, tensor)
#
#     mean = numpy.mean(means, dtype=numpy.float64)
#     var = numpy.mean((vars + (means - mean) ** 2))
#     mean = float(mean)
#     std = float(numpy.sqrt(var))
#
#     assert len(min_.shape) == 2, min_.shape
#     assert len(max_.shape) == 2, max_.shape
#     return min_, max_, mean, std


def get_min_max_mean_std(tensor: numpy.ndarray):
    min_ = numpy.min(tensor, axis=0)
    assert len(min_.shape) == 2, min_.shape
    max_ = numpy.max(tensor, axis=0)
    assert len(max_.shape) == 2, max_.shape
    mean = numpy.mean(tensor, axis=0)
    assert len(mean.shape) == 2, mean.shape
    std = numpy.std(tensor, axis=0)
    assert len(std.shape) == 2, std.shape

    return min_, max_, mean, std


def trace_peaks(tensor: numpy.ndarray, name: str, peaks: numpy.ndarray, reduce: str):
    assert len(peaks.shape) == 2, peaks.shape
    P = peaks.shape[0]
    assert peaks.shape[1] == 3, peaks.shape

    T, H, W = tensor.shape
    peak_masks = numpy.stack([create_circular_mask(H, W, p).flatten() for p in peaks], axis=1)
    cirlce_area = peak_masks[:, 0].sum()
    assert len(peak_masks.shape) == 2, peak_masks.shape
    assert peak_masks.shape[0] == H * W, (peak_masks.shape, H, W)
    assert peak_masks.shape[1] == P, (peak_masks.shape, P)

    tensor = tensor.reshape(T, H * W)

    if reduce == "mean":
        traces = tensor.dot(peak_masks) / cirlce_area
    elif reduce == "max":
        # masked_tensor = numpy.multiply(tensor[..., None], peak_masks[None, ...])
        # traces = numpy.max(masked_tensor, axis=1)
        traces = numpy.max(numpy.repeat(tensor[..., None], P, axis=-1), axis=1, where=peak_masks[None, ...])
    else:
        raise NotImplementedError(reduce)

    traces = traces.T
    assert traces.shape == (P, T), (traces.shape, (P, T))
    return traces


def create_circular_mask(h: int, w: int, peak: Tuple[int, int, int]):
    center = peak[:2]
    radius = peak[2]
    H, W = numpy.ogrid[:h, :w]
    dist_from_center = numpy.sqrt((H - center[0]) ** 2 + (W - center[1]) ** 2)

    mask = dist_from_center <= radius
    return mask


def get_smooth_traces(traces, wname, w):
    if isinstance(w, str):
        return getattr(scipy.signal, wname)(traces, **json.loads(w))
    else:
        window = scipy.signal.get_window(wname, w, fftbins=False)
        window /= window.sum()
        return numpy.stack([numpy.convolve(window, trace, mode="valid") for trace in traces])


def get_smooth_traces_pair(
    recon: str,
    smooth: Optional[Tuple[str, Union[int, dict]]],
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
        if isinstance(w, str):
            w = 0

    if tgt_smooth is None:
        tgt_traces = all_traces[tgt]
        tgt_w = 0
    else:
        tgt_wname, tgt_w = tgt_smooth
        smooth_tgt_traces = all_smooth_traces.get((tgt, tgt_wname, tgt_w), None)
        if smooth_tgt_traces is None:
            smooth_tgt_traces = get_smooth_traces(all_traces[tgt], tgt_wname, tgt_w)
            all_smooth_traces[tgt, tgt_wname, tgt_w] = smooth_tgt_traces

        tgt_traces = smooth_tgt_traces
        if isinstance(tgt_w, str):
            tgt_w = 0

    if w < tgt_w:
        wdiff = tgt_w - w
        traces = traces[:, wdiff // 2 : -(wdiff // 2)]
    elif tgt_w < w:
        wdiff = w - tgt_w
        tgt_traces = tgt_traces[:, wdiff // 2 : -(wdiff // 2)]

    return traces, tgt_traces


def plot_traces(*, tgt, plots, all_traces, all_smooth_traces, correlations, output_path, tag: str = ""):
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
        return ax.plot(*xy, **kwargs), y_center, half_range

    def plot_trace(
        ax,
        j,
        traces,
        t,
        name,
        *,
        i=None,
        wname="",
        w: Union[str, int] = "",
        w_max: int = 0,
        corr_key=None,
        y_center_and_half_range: Optional[Tuple] = None,
        tag: str = "",  # for plot title only
    ):
        label = f"{trace_name_map.get(name, name):>3} {wname:>6} {w:<9} "
        label = label.replace("{", "")
        label = label.replace("}", "")
        label = label.replace('"', "")
        label = label.replace("polyorder", "p")
        label = label.replace("window_length", "w")
        label = label.replace("_filter", "")
        label = label.replace(", ", " ")
        if corr_key in correlations:
            label += (
                f"Pears={correlations[corr_key]['pearson']:.3f}  " f"Spear={correlations[corr_key]['spearman']:.3f}"
            )
        plot_args_here = [numpy.arange(w_max // 2, all_traces[tgt].shape[1] - w_max // 2), traces[t]]
        plot_kwargs_here = {"label": label, "color": get_color(i, j), **plot_kwargs}
        if y_center_and_half_range is None:
            return plot_centered(*plot_args_here, ax=ax, **plot_kwargs_here)
        else:
            return plot_centered(*plot_args_here, ax=ax, **plot_kwargs_here)[0]
            # y_center, half_range = y_center_and_half_range
            # ax.set_ylim(y_center - half_range, y_center + half_range)
            # return ax.plot(*plot_args_here, **plot_kwargs_here)

    # from https://matplotlib.org/3.1.1/gallery/ticks_and_spines/multiple_yaxis_with_spines.html
    def make_patch_spines_invisible(ax):
        ax.set_frame_on(True)
        ax.patch.set_visible(False)
        for sp in ax.spines.values():
            sp.set_visible(False)

    fontdict = {"family": "monospace"}
    figs = {}
    rel_dist_per_recon = 0.04
    for t in range(all_traces[tgt].shape[0]):
        nrows = len(plots)
        fig, axes = plt.subplots(nrows=nrows, sharex=True, figsize=(20, 10), squeeze=False)
        axes = axes[:, 0]  # only squeeze ncols=1
        # plt.suptitle(f"Trace {i:2}")
        axes[0].set_title(f"{tag} Trace {t:2}")
        for ax in axes:
            ax.tick_params(axis="y", labelcolor=get_color(None, 0))
            ax.set_xlim(0, all_traces[tgt].shape[1])
            ax.set_ylabel(trace_name_map.get(tgt, tgt), color=get_color(None, 0), fontdict=fontdict)

        i = 0
        for ax, recons in zip(axes, plots):
            plotted_lines = []
            for i, (recon, kwargs) in enumerate(recons.items()):
                j = 0
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

                    w_max = max(0 if isinstance(w, str) else w, 0 if isinstance(tgt_w, str) else tgt_w)
                    pl, y_center, half_range = plot_trace(
                        ax, j, tgt_traces, t, tgt, wname=tgt_wname, w=tgt_w, w_max=w_max, tag=tag
                    )
                    plotted_lines += pl
                    plotted_lines += plot_trace(
                        twinx,
                        j,
                        traces,
                        t,
                        recon,
                        i=i,
                        wname=wname,
                        w=w,
                        w_max=w_max,
                        corr_key=(recon, smooth, tgt_smooth, t),
                        y_center_and_half_range=(y_center, half_range),
                        tag=tag,
                    )
                    j += 1

            labels = [l.get_label() for l in plotted_lines]
            ax.legend(
                plotted_lines,
                labels,
                bbox_to_anchor=(1.0 + rel_dist_per_recon * (i + 1), 0.5),
                loc="center left",
                # bbox_to_anchor=(.5 , 1.05),
                # loc="center top",
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


def add_paths_to_plots(plots, paths):
    for plot in plots:
        for recon, kwargs in plot.items():
            kwargs["path"] = paths[recon]
            kwargs["smooth"] = [
                tuple(
                    [
                        None
                        if smoo is None
                        else tuple([json.dumps(sm, sort_keys=True) if isinstance(sm, dict) else sm for sm in smoo])
                        for smoo in smooth
                    ]
                )
                for smooth in kwargs["smooth"]
            ]

    return plots


if __name__ == "__main__":
    paths_09_3_a = {
        330: {
            "lr_slice": Path(
                "/g/kreshuk/LF_computed/lnet/logs/brain1/test_z_out49/lr_f4/20-05-31_21-24-03/brain.09_3__2020-03-09_06.43.40__SinglePlane_-330/run000/ds0-0"
            ),
            "ls_slice": Path(
                "/g/kreshuk/LF_computed/lnet/logs/brain1/test_z_out49/lr_f4/20-05-31_21-24-03/brain.09_3__2020-03-09_06.43.40__SinglePlane_-330/run000/ds0-0"
            ),
            "pred": Path(
                "/g/kreshuk/LF_computed/lnet/logs/brain1/test_z_out49/f4/z_out49/f4_b2_only09_3/20-05-30_10-41-55/v1_checkpoint_82000_MSSSIM=0.8523718668864324/brain.09_3__2020-03-09_06.43.40__SinglePlane_-330/run000/ds0-0/"
            ),
        }
    }

    paths_11_2 = {
        290: {
            "ls_slice": Path(
                f"/g/kreshuk/LF_computed/lnet/logs/brain1/test_z_out49/lr_f4/20-06-10_16-36-00/brain.11_2__2020-03-11_10.13.20__SinglePlane_-290/run000/ds0-0"
            ),
            "lr_slice": Path(
                f"/g/kreshuk/LF_computed/lnet/logs/brain1/test_z_out49/lr_f4/20-06-10_16-36-00/brain.11_2__2020-03-11_10.13.20__SinglePlane_-290/run000/ds0-0"
            ),
            "pred": Path(
                f"/g/kreshuk/LF_computed/lnet/logs/brain1/test_z_out49/f4/z_out49/f4_b2_only11_2/20-06-06_17-59-42/v1_checkpoint_29500_MS_SSIM=0.8786535175641378/brain.11_2__2020-03-11_10.13.20__SinglePlane_-290/run000/ds0-0"
            ),
        },
        295: {
            "ls_slice": Path(
                f"/g/kreshuk/LF_computed/lnet/logs/brain1/test_z_out49/lr_f4/20-06-10_16-29-33/brain.11_2__2020-03-11_10.25.41__SinglePlane_-295/run000/ds0-0"
            ),
            "lr_slice": Path(
                f"/g/kreshuk/LF_computed/lnet/logs/brain1/test_z_out49/lr_f4/20-06-10_16-29-33/brain.11_2__2020-03-11_10.25.41__SinglePlane_-295/run000/ds0-0"
            ),
            "pred": Path(
                f"/g/kreshuk/LF_computed/lnet/logs/brain1/test_z_out49/f4/z_out49/f4_b2_only11_2/20-06-06_17-59-42/v1_checkpoint_29500_MS_SSIM=0.8786535175641378/brain.11_2__2020-03-11_10.25.41__SinglePlane_-295/run000/ds0-0"
            ),
        },
        310: {
            "ls_slice": Path(
                f"/g/kreshuk/LF_computed/lnet/logs/brain1/test_z_out49/lr_f4/20-05-31_21-24-03/brain.11_2__2020-03-11_07.30.39__SinglePlane_-310/run000/ds0-0"
            ),
            "lr_slice": Path(
                f"/g/kreshuk/LF_computed/lnet/logs/brain1/test_z_out49/lr_f4/20-05-31_21-24-03/brain.11_2__2020-03-11_07.30.39__SinglePlane_-310/run000/ds0-0"
            ),
            "pred": Path(
                f"/g/kreshuk/LF_computed/lnet/logs/brain1/test_z_out49/f4/z_out49/f4_b2_only11_2/20-06-06_17-59-42/v1_checkpoint_29500_MS_SSIM=0.8786535175641378/brain.11_2__2020-03-11_07.30.39__SinglePlane_-310/run000/ds0-0"
            ),
        },
        320: {
            "ls_slice": Path(
                f"/g/kreshuk/LF_computed/lnet/logs/brain1/test_z_out49/lr_f4/20-05-31_21-24-03/brain.11_2__2020-03-11_07.30.39__SinglePlane_-320/run000/ds0-0"
            ),
            "lr_slice": Path(
                f"/g/kreshuk/LF_computed/lnet/logs/brain1/test_z_out49/lr_f4/20-05-31_21-24-03/brain.11_2__2020-03-11_07.30.39__SinglePlane_-320/run000/ds0-0"
            ),
            "pred": Path(
                f"/g/kreshuk/LF_computed/lnet/logs/brain1/test_z_out49/f4/z_out49/f4_b2_only11_2/20-06-06_17-59-42/v1_checkpoint_29500_MS_SSIM=0.8786535175641378/brain.11_2__2020-03-11_07.30.39__SinglePlane_-320/run000/ds0-0"
            ),
        },
        330: {
            "ls_slice": Path(
                f"/g/kreshuk/LF_computed/lnet/logs/brain1/test_z_out49/lr_f4/20-05-31_21-24-03/brain.11_2__2020-03-11_06.53.14__SinglePlane_-330/run000/ds0-0"
            ),
            "lr_slice": Path(
                f"/g/kreshuk/LF_computed/lnet/logs/brain1/test_z_out49/lr_f4/20-05-31_21-24-03/brain.11_2__2020-03-11_06.53.14__SinglePlane_-330/run000/ds0-0"
            ),
            "pred": Path(
                f"/g/kreshuk/LF_computed/lnet/logs/brain1/test_z_out49/f4/z_out49/f4_b2_only11_2/20-06-06_17-59-42/v1_checkpoint_29500_MS_SSIM=0.8786535175641378/brain.11_2__2020-03-11_06.53.14__SinglePlane_-330/run000/ds0-0"
            ),
        },
    }

    for tag in [
        "11_2__2020-03-11_06.53.14__SinglePlane_-330",
        "11_2__2020-03-11_07.30.39__SinglePlane_-310",
        "11_2__2020-03-11_07.30.39__SinglePlane_-320",
        "11_2__2020-03-11_10.13.20__SinglePlane_-290",
        "11_2__2020-03-11_10.17.34__SinglePlane_-280",
        "11_2__2020-03-11_10.17.34__SinglePlane_-330",
        "11_2__2020-03-11_10.21.14__SinglePlane_-295",
        "11_2__2020-03-11_10.21.14__SinglePlane_-305",
        "11_2__2020-03-11_10.25.41__SinglePlane_-295",
        "11_2__2020-03-11_10.25.41__SinglePlane_-340",
    ]:
        paths_11_2[tag] = {
            name: Path(
                f"/g/kreshuk/LF_computed/lnet/logs/brain1/test_z_out49/lr_f4/20-06-12_22-07-43/brain.{tag}/run000/ds0-0"
            )
            for name in ["ls_slice", "lr_slice"]
        }

    for i, tag in enumerate(
        [
            # "11_2__2020-03-11_06.53.14__SinglePlane_-330",
            "11_2__2020-03-11_07.30.39__SinglePlane_-310",
            # "11_2__2020-03-11_07.30.39__SinglePlane_-320",
            # "11_2__2020-03-11_10.13.20__SinglePlane_-290",
            # "11_2__2020-03-11_10.17.34__SinglePlane_-280",
            # "11_2__2020-03-11_10.17.34__SinglePlane_-330",
            # "11_2__2020-03-11_10.21.14__SinglePlane_-295",
            # "11_2__2020-03-11_10.21.14__SinglePlane_-305",
            # "11_2__2020-03-11_10.25.41__SinglePlane_-295",
            # "11_2__2020-03-11_10.25.41__SinglePlane_-340",
        ]
    ):
        paths = paths_11_2[tag]
        # paths = paths_09_3_a[330]
        output_path = Path(f"/g/kreshuk/LF_computed/lnet/trace_debug_motion{i}")
        tgt = "ls_slice"
        plots = add_paths_to_plots(
            [
                # {"lr_slice": {"smooth": [(None, None)]}, "pred": {"smooth": [(None, None)]}},
                # {"lr_slice": {"smooth": [(("flat", 5), ("flat", 7))]}, "pred": {"smooth": [(("flat", 3), ("flat", 5))]}},
                # {"lr_slice": {"smooth": [(None, ("flat", 3))]}, "pred": {"smooth": [(None, ("flat", 3))]}},
                {"lr_slice": {"smooth": [(None, None)]}},
                {"lr_slice": {"smooth": [(("flat", 11), ("flat", 11))]}},
                # {"lr_slice": {"smooth": [(None, ("flat", 3))]}},
                {
                    "lr_slice": {
                        "smooth": [
                            (
                                ("savgol_filter", {"window_length": 11, "polyorder": 3}),
                                ("savgol_filter", {"window_length": 11, "polyorder": 3}),
                            )
                        ]
                    },
                    # "pred": {
                    #     "smooth": [
                    #         (
                    #             ("savgol_filter", {"window_length": 11, "polyorder": 3}),
                    #             ("savgol_filter", {"window_length": 11, "polyorder": 3}),
                    #         )
                    #     ]
                    # },
                },
                # {
                #     "lr_slice": {
                #         "smooth": [
                #             (
                #                 ("savgol_filter", {"window_length": 9, "polyorder": 5}),
                #                 ("savgol_filter", {"window_length": 9, "polyorder": 5}),
                #             )
                #         ]
                #     },
                #     "pred": {
                #         "smooth": [
                #             (
                #                 ("savgol_filter", {"window_length": 9, "polyorder": 3}),
                #                 ("savgol_filter", {"window_length": 9, "polyorder": 3}),
                #             )
                #         ]
                #     },
                # },
            ],
            paths=paths,
        )

        peaks, traces, correlations, figs, motion = trace(
            tgt_path=paths[tgt],
            tgt=tgt,
            roi=(slice(0, 9999), slice(0, 9999)),
            # roi=(slice(5, 25), slice(5, 25)),
            plots=plots,
            output_path=output_path,
            nr_traces=1,
            overwrite_existing_files=True,
            smooth_diff_sigma=1.3,
            peak_threshold_abs=0.05,
            reduce_peak_area="mean",
            plot_peaks=True,
            compute_peaks_on="std",  # std, diff
            peaks_min_dist=3,
            trace_radius=2,
            # time_range=(0, 600),
            # time_range=(0, 50),
            # time_range=(660, 1200),
            time_range=(20, None),
            # compensate_motion={"compensate_ref": tgt, "method": "DS", "mbSize": 50, "p": 4},
            tag=tag,
        )

        for name, trace in traces.items():
            print("trace", name, trace.shape, trace.min(), trace.max())

        plt.show()

# todo:
# LR: whta's there?
# more predictions
#
