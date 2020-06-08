from pathlib import Path
from typing import Dict, Optional, Set, Tuple, Union

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
    compare_to: Union[Dict[str, Optional[Path]], Set[str]],
    overwrite_existing_files: bool = False,
):
    if isinstance(compare_to, set):
        compare_to = {ct: tgt_path for ct in compare_to}
    elif isinstance(compare_to, dict):
        compare_to = {ct: p or tgt_path for ct, p in compare_to.items()}
    else:
        raise TypeError(type(compare_to))

    ds_tgt = TiffDataset(info=TensorInfo(name=tgt, root=tgt_path, location=f"{tgt}/*.tif"))
    length = len(ds_tgt)
    datasets_to_trace = {
        ct: TiffDataset(info=TensorInfo(name=ct, root=p, location=f"{ct}/*.tif")) for ct, p in compare_to.items()
    }
    assert all([len(ctd) == length for ctd in datasets_to_trace.values()]), [length] + [
        len(ctd) for ctd in datasets_to_trace.values()
    ]
    assert tgt not in datasets_to_trace
    datasets_to_trace[tgt] = ds_tgt

    figs = {}
    peak_path = tgt_path / f"{tgt}_peaks.yml"
    if peak_path.exists() and not overwrite_existing_files:
        with peak_path.open() as f:
            peaks = numpy.asarray(yaml.safe_load(f))
    else:
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
        peaks = peak_local_max(smooth_diff_tensor, min_distance=3, threshold_abs=1.0, exclude_border=True, num_peaks=1)
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

    traces = {}
    assert tgt not in compare_to
    for ct, p in {tgt: tgt_path, **compare_to}.items():
        traces_path: Path = p / f"{ct}_traces.npy"
        if traces_path.exists() and not overwrite_existing_files:
            traces[ct] = numpy.load(str(traces_path))
        else:
            traces[ct] = trace_peaks(datasets_to_trace[ct], ct, peaks)
            numpy.save(str(traces_path), traces[ct])

    smooth_with = [("hann", 5), ("hann", 3), ("flat", 3)]
    smooth_traces = {}
    for wname, w in smooth_with:
        window = scipy.signal.get_window(wname, w, fftbins=False)
        window /= window.sum()
        for name, trace in traces.items():
            if name == "ls_slice":  # todo: remove this hack
                smooth_trace = numpy.stack([numpy.convolve(window, t, mode="valid") for t in trace])
            else:
                smooth_trace = trace[:, w // 2 : -(w // 2)]

            smooth_traces[name, wname, w] = smooth_trace

    tgt_trace = traces.pop(tgt)
    smooth_tgt_traces = {}
    for name, wname, w in smooth_traces:
        if name == tgt:
            smooth_tgt_traces[wname, w] = smooth_traces.pop((name, wname, w))


    correlations = {}
    for key, trace in {**traces, **smooth_traces}.items():
        pearson_corr = []
        spearman_corr = []
        if isinstance(key, tuple):
            name, wname, w = key
            tgt_trace = smooth_tgt_traces[wname, w]
        else:
            name = key

        for t, tgt_t in zip(trace, tgt_trace):
            pear_corr, _ = pearsonr(t, tgt_t)
            pearson_corr.append(pear_corr)
            spear_corr, _ = spearmanr(t, tgt_t)
            spearman_corr.append(spear_corr)

        correlations[name] = {"pearson": pearson_corr, "spearman": spearman_corr}

    for (name, wname, w), trace in smooth_traces.items():
        pearson_corr = []
        spearman_corr = []
        for t, tgt_t in zip(trace, smooth_tgt_trace):
            pear_corr, _ = pearsonr(t, tgt_t)
            pearson_corr.append(pear_corr)
            spear_corr, _ = spearmanr(t, tgt_t)
            spearman_corr.append(spear_corr)

            correlations[(name, wname, w)] = {"pearson": pearson_corr, "spearman": spearman_corr}

    trace_figs = plot_traces(
        tgt=tgt,
        tgt_trace=tgt_trace,
        compare_to=compare_to,
        traces=traces,
        smooth_with=smooth_with,
        smooth_tgt_traces=smooth_tgt_traces,
        all_smooth_traces=all_smooth_traces,
        correlations=correlations,
    )
    figs.update(trace_figs)
    return peaks, traces, correlations, figs


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


def plot_traces(
    *,
    tgt,
    tgt_trace,
    compare_to,
    traces,
    smooth_with,
    smooth_tgt_traces,
    all_smooth_traces,
    correlations,
    individual_subfigures=True,
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

    def get_color(i: Optional[int], j):
        n = len(smooth_with)
        if i is None:
            return tgt_cmap(j / n * 0.75)
        else:
            return sequential_cmaps[i](j / n * 0.75)

    plot_kwargs = {"linewidth": 1}

    def plot_centered(*xy, ax, **kwargs):
        y: numpy.ndarray = xy[-1]
        y_min = numpy.min(y)
        y_max = numpy.max(y)
        y_center = numpy.median(y)
        half_range = max(abs(y_center - y_min), abs(y_center - y_max))
        ax.set_ylim(y_center - half_range, y_center + half_range)
        return ax.plot(*xy, **kwargs)

    def get_trace_label(name, wname, w):
        label = f"{trace_name_map.get(name, name):>3} {wname:>5} {w:<1} "
        if wname:
            label += (
                f"Pearson: {correlations[name, wname, w]['pearson'][t]:.3f}  "
                f"Spearman: {correlations[name, wname, w]['spearman'][t]:.3f}"
            )
        elif name in correlations:
            label += (
                f"Pearson: {correlations[name]['pearson'][t]:.3f}  "
                f"Spearman: {correlations[name]['spearman'][t]:.3f}"
            )

        return label

    def plot_trace(ax, j, trace, name, *, i=None, wname="", w=0):
        label = get_trace_label(name, wname, w)
        return plot_centered(
            numpy.arange(w // 2, tgt_trace.shape[1] - w // 2),
            trace,
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
    for t in range(tgt_trace.shape[0]):
        if individual_subfigures:
            nrows = len(smooth_with) + 1
        else:
            nrows = 2

        fig, axes = plt.subplots(nrows=nrows, sharex=True, figsize=(20, 4))
        # plt.suptitle(f"Trace {i:2}")
        axes[0].set_title(f"Trace {t:2}")
        for ax in axes:
            ax.tick_params(axis="y", labelcolor=get_color(None, 0))
            ax.set_xlim(0, tgt_trace.shape[1])
            ax.set_ylabel(trace_name_map.get(tgt, tgt), color=get_color(None, 0), fontdict=fontdict)

        plotted_lines = []
        plotted_lines.append(plot_trace(axes[0], 0, tgt_trace[t], tgt))  # plot tgt trace to first subfig

        wname, w = smooth_with[0]
        plotted_lines.append(plot_trace(axes[1], 1, smooth_tgt_traces[wname, w][t], tgt, wname=wname, w=w))
        for j, (wname, w) in enumerate(smooth_with[1:], start=2):
            axj = j if individual_subfigures else 1
            plotted_lines[axj] += plot_trace(axes[axj], j, smooth_tgt_traces[wname, w][t], tgt, wname=wname, w=w)

        rel_dist_per_recon = 0.03
        i = 0
        for i, name in enumerate(traces):
            twinx_axes = [ax.twinx() for ax in axes]  # new twinx for each recon (pred, lr_slice, ...)
            for twinx in twinx_axes:
                # twinx.set_ylabel(trace_name_map.get(name, name), fontdict=fontdict)
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

            plotted_lines[0] += plot_trace(
                twinx_axes[0], 0, traces[name][t], name, i=i
            )  # plot unsmoothed recon to first subfig
            for j, (wname, w) in enumerate(smooth_with, start=1):
                axj = j if individual_subfigures else 1
                # twinx_axes[axj].set_ylabel(trace_name_map.get(name, name), fontdict=fontdict)
                plotted_lines[axj] += plot_trace(
                    twinx_axes[axj], j, all_smooth_traces[wname, w][t], name, i=i, wname=wname, w=w
                )

        for ax, lns in zip(axes, plotted_lines):
            labels = [l.get_label() for l in lns]
            ax.legend(
                lns, labels, bbox_to_anchor=(1.0 + rel_dist_per_recon * (i + 1), 0.5), loc="center left", prop=fontdict
            )

        figs[f"trace{t}"] = fig
        for name in traces:
            plt.savefig(compare_to[name] / f"trace{t}.svg")

    return figs


if __name__ == "__main__":
    # peaks, traces, correlations, figs = trace(
    #     # tgt_path = Path("/g/kreshuk/LF_computed/lnet/logs/brain1/z_out49/f2_only11_2/20-05-19_12-27-16/test/run000/ds0-0")
    #     tgt_path=Path(
    #         "/g/kreshuk/LF_computed/lnet/logs/brain1/test_z_out49/lr_f4/20-05-31_21-24-03/brain.09_3__2020-03-09_06.43.40__SinglePlane_-330/run000/ds0-0"
    #     ),
    #     tgt="ls_slice",
    #     compare_to={"lr_slice"},
    # )
    # for name, trace in traces.items():
    #     print(name, trace.shape, trace.min(), trace.max())
    #
    # plt.show()
    peaks, traces, correlations, figs = trace(
        tgt_path=Path(
            "/g/kreshuk/LF_computed/lnet/logs/brain1/test_z_out49/lr_f4/20-05-31_21-24-03/brain.09_3__2020-03-09_06.43.40__SinglePlane_-330/run000/ds0-0"
        ),
        tgt="ls_slice",
        compare_to={
            "pred": Path(
                "/g/kreshuk/LF_computed/lnet/logs/brain1/test_z_out49/f4/z_out49/f4_b2_only09_3/20-05-30_10-41-55/v1_checkpoint_82000_MSSSIM=0.8523718668864324/brain.09_3__2020-03-09_06.43.40__SinglePlane_-330/run000/ds0-0/"
            ),
            "lr_slice": Path(
                "/g/kreshuk/LF_computed/lnet/logs/brain1/test_z_out49/lr_f4/20-05-31_21-24-03/brain.09_3__2020-03-09_06.43.40__SinglePlane_-330/run000/ds0-0"
            ),
        },
    )
    for name, trace in traces.items():
        print(name, trace.shape, trace.min(), trace.max())

    plt.show()

    # todo: all compare_to in same plot
    # todo: plot symmetrically around mean
