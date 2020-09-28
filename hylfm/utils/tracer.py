import json
import logging
import math
import warnings
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
from hashlib import sha256
from itertools import accumulate
from pathlib import Path
from pprint import pprint
from typing import Dict, List, Optional, Set, Tuple, Union

import matplotlib.pyplot as plt
import numpy
import scipy.signal
import skimage.transform
import tifffile
from imageio import imread, imwrite
from ruamel.yaml import YAML
from scipy.ndimage import gaussian_filter
from scipy.stats import pearsonr, spearmanr
from skimage.feature import peak_local_max
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from hylfm import settings
from hylfm.datasets import TensorInfo
from hylfm.datasets.base import TiffDataset, get_collate_fn
from hylfm.utils.general import print_timing

plt.rcParams["svg.fonttype"] = "none"

try:
    import skvideo.motion
except ImportError:
    warnings.warn("could not import skvideo")

logger = logging.getLogger(__name__)
yaml = YAML(typ="safe")

SHOW_FIGS = True

# use tifffile instead of imageio, because imageio.volwrite(p, data) throws an exception when passing 'compress' kwarg
def volwrite(p: Path, data, compress=2, **kwargs):
    with p.open("wb") as f:
        tifffile.imsave(f, data, compress=compress, **kwargs)


@print_timing
def trace_and_plot(
    tgt_path: Union[str, Path],
    tgt: str,
    roi: Tuple[slice, slice],
    plots: List[Union[Dict[str, Dict[str, Union[str, Path, List]]], Set[str]]],
    output_path: Path,
    nr_traces: int,
    background_threshold: Optional[float] = None,
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
    peak_path: Optional[Union[str, Path]] = None,
    compensated_peak_path: Optional[Union[str, Path]] = None,
):
    for plot in plots:
        for recon, kwargs in plot.items():
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

    if isinstance(tgt_path, str):
        tgt_path = Path(tgt_path)

    if isinstance(compensated_peak_path, str):
        compensated_peak_path = Path(compensated_peak_path)

    def path2string(obj):
        if isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, list):
            return [path2string(v) for v in obj]
        elif isinstance(obj, set):
            return {path2string(v) for v in obj}
        elif isinstance(obj, dict):
            return {k: path2string(v) for k, v in obj.items()}
        else:
            return obj

    all_kwargs = {
        "tgt_path": str(tgt_path),
        "tgt": tgt,
        "roi": str(roi),
        "plots": path2string(plots),
        "nr_traces": nr_traces,
        "background_threshold": background_threshold,
        "overwrite_existing_files": overwrite_existing_files,
        "smooth_diff_sigma": smooth_diff_sigma,
        "peak_threshold_abs": peak_threshold_abs,
        "reduce_peak_area": reduce_peak_area,
        "time_range": time_range,
        "plot_peaks": plot_peaks,
        "compute_peaks_on": compute_peaks_on,
        "peaks_min_dist": peaks_min_dist,
        "trace_radius": trace_radius,
        "compensate_motion": compensate_motion,
        "tag": tag,
        "peak_path": None if peak_path is None else str(peak_path),
        "compensated_peak_path": None if compensated_peak_path is None else str(compensated_peak_path),
    }
    descr_hash = sha256()
    descr_hash.update(json.dumps(all_kwargs, sort_keys=True).encode("utf-8"))
    output_path /= descr_hash.hexdigest()
    print("output path:", output_path)
    output_path.mkdir(exist_ok=True, parents=True)
    yaml.dump(all_kwargs, output_path / "kwargs.yaml")
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
            assert len(datasets_to_trace[name]) <= 600, "accross TP??"

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

        # plt.imshow(datasets_to_trace[name].max(axis=0))
        # plt.title(name)
        # plt.show()

    compensate_motion_of_peaks = compensate_motion is not None and compensate_motion.pop("of_peaks", False)
    if not compensate_motion_of_peaks and compensate_motion is not None:
        compensate_ref_name = compensate_motion.pop("compensate_ref", tgt)
        assert compensate_ref_name == tgt, "not implemented"
        compensate_ref = datasets_to_trace[compensate_ref_name]
        motion = skvideo.motion.blockMotion(compensate_ref, **compensate_motion)
        # print("motion", motion.shape, motion.max())
        assert numpy.isfinite(motion).all()
        for name in set(datasets_to_trace.keys()):
            data = datasets_to_trace[name]
            # print("data", data.shape)

            # compensate the video
            compensate = (
                skvideo.motion.blockComp(data, motion, mbSize=compensate_motion.get("mbSize", 8))
                .squeeze(axis=-1)
                .astype("float32")
            )
            # print("compensate", compensate.shape)
            assert numpy.isfinite(compensate).all()
            # write
            volwrite(output_path / f"{name}_motion_compensated.tif", compensate)
            volwrite(output_path / f"{name}_not_compensated.tif", data)
    else:
        motion = None

    compensated_peaks = None
    figs = {}
    if peak_path is None and compensated_peak_path is None:
        peak_path = output_path / f"{tgt}_peaks_of_{compute_peaks_on}.yml"
        peaks = None
        if peak_path.exists() and not overwrite_existing_files:
            peaks = numpy.asarray(yaml.load(peak_path))

            if peaks.shape[0] != nr_traces:
                peaks = None
    elif peak_path is None and compensated_peak_path is not None:
        assert compensated_peak_path.exists()
        compensated_peaks = numpy.asarray(yaml.load(compensated_peak_path)).T[None, ...]
        assert nr_traces == 1
    else:
        assert peak_path.exists()
        peaks = numpy.asarray(yaml.load(peak_path))
        nr_traces = min(nr_traces, peaks.shape[0])

    if compensated_peaks is None and (peaks is None or plot_peaks):
        all_projections = OrderedDict()
        for tensor_name in {tgt, *all_recon_paths.keys()}:
            min_path = output_path / f"{tensor_name}_min.tif"
            max_path = output_path / f"{tensor_name}_max.tif"
            mean_path = output_path / f"{tensor_name}_mean.tif"
            std_path = output_path / f"{tensor_name}_std.tif"

            if (
                min_path.exists()
                and max_path.exists()
                and mean_path.exists()
                and std_path.exists()
                and not overwrite_existing_files
            ):
                min_tensor = imread(min_path)
                max_tensor = imread(max_path)
                mean_tensor = imread(mean_path)
                std_tensor = imread(std_path)
            else:
                min_tensor, max_tensor, mean_tensor, std_tensor = get_min_max_mean_std(datasets_to_trace[tensor_name])
                imwrite(min_path, min_tensor)
                imwrite(max_path, max_tensor)
                imwrite(mean_path, mean_tensor)
                imwrite(std_path, std_tensor)

            all_projections[tensor_name] = {
                "min": min_tensor,
                "max": max_tensor,
                "mean": mean_tensor,
                "std": std_tensor,
            }

        all_projections.move_to_end(tgt, last=False)
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

            def get_peaks_on_tensor(comp_on: str):
                if comp_on in ["min", "max", "std", "mean"]:
                    peaks_on_tensor = projections[comp_on]
                elif comp_on == "diff":
                    return diff_tensor
                elif comp_on == "smooth_diff":
                    peaks_on_tensor = gaussian_filter(diff_tensor, sigma=smooth_diff_sigma, mode="constant")
                    plot_peaks_on["smooth diff tensor"] = peaks_on_tensor
                elif "+" in comp_on:
                    comp_on_parts = comp_on.split("+")
                    peaks_on_tensor = get_peaks_on_tensor(comp_on_parts[0])
                    for part in comp_on_parts[1:]:
                        peaks_on_tensor = numpy.add(peaks_on_tensor, get_peaks_on_tensor(part))
                elif "*" in comp_on:
                    factor, comp_on = comp_on.split("*")
                    factor = float(factor)
                    peaks_on_tensor = factor * get_peaks_on_tensor(comp_on)
                else:
                    raise NotImplementedError(compute_peaks_on)

                return peaks_on_tensor

            peaks_on_tensor = get_peaks_on_tensor(compute_peaks_on)
            background_mask = (
                None
                if background_threshold is None
                else (all_projections[tgt]["min"] > background_threshold).astype(numpy.int)
            )
            imwrite(output_path / "background_mask.tif", background_mask.astype("uint8") * 255)
            if tensor_name == tgt:
                peaks = peak_local_max(
                    peaks_on_tensor,
                    min_distance=peaks_min_dist,
                    threshold_abs=peak_threshold_abs,
                    exclude_border=True,
                    num_peaks=nr_traces,
                    labels=background_mask,
                )
                peaks = numpy.concatenate([peaks, numpy.full((peaks.shape[0], 1), trace_radius)], axis=1)

            # plot peak positions on different projections
            fig, axes = plt.subplots(nrows=math.ceil(len(plot_peaks_on) / 3), ncols=3, squeeze=False, figsize=(15, 10))
            plt.suptitle(tensor_name)
            [ax.set_axis_off() for ax in axes.flatten()]
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

            try:
                plt.tight_layout()
            except Exception as e:
                warnings.warn(e)

            fig_name = f"trace_positions_on_{tensor_name}"
            plt.savefig(output_path / f"{fig_name}.svg")
            plt.savefig(output_path / f"{fig_name}.png")
            figs[fig_name] = fig
            if SHOW_FIGS:
                plt.show()
            else:
                plt.close()

            yaml.dump(peaks.tolist(), peak_path)

    if compensated_peaks is not None:
        peaks = None
        all_compensated_peaks = {tgt: compensated_peaks}
        for name in all_recon_paths.keys():
            all_compensated_peaks[name] = compensated_peaks
    elif compensate_motion_of_peaks:
        only_on_tgt = compensate_motion.pop("only_on_tgt", False)
        print(f"get_motion_compensated_peaks for {tgt}")
        all_compensated_peaks = {
            tgt: get_motion_compensated_peaks(
                tensor=datasets_to_trace[tgt], peaks=peaks, output_path=output_path, **compensate_motion, name=tgt
            )
        }
        for name in {*all_recon_paths.keys()}:
            if only_on_tgt:
                all_compensated_peaks[name] = all_compensated_peaks[tgt]
            else:
                print(f"get_motion_compensated_peaks for {name}")
                all_compensated_peaks[name] = get_motion_compensated_peaks(
                    tensor=datasets_to_trace[name], peaks=peaks, output_path=output_path, **compensate_motion, name=name
                )
    else:
        all_compensated_peaks = None

    all_traces = {}
    if all_compensated_peaks is None:
        for name in {tgt, *all_recon_paths.keys()}:
            traces_path: Path = output_path / f"{name}_traces.npy"
            if traces_path.exists() and not overwrite_existing_files:
                all_traces[name] = numpy.load(str(traces_path))
            else:
                all_traces[name] = trace_straight_peaks(datasets_to_trace[name], peaks, reduce_peak_area, output_path)
                numpy.save(str(traces_path), all_traces[name])
    else:
        for name, compensated_peaks in all_compensated_peaks.items():
            # traces_path: Path = output_path / f"{name}_traces.npy"
            # if traces_path.exists() and not overwrite_existing_files:
            #     all_traces[name] = numpy.load(str(traces_path))
            # else:
            all_traces[name] = trace_tracked_peaks(
                datasets_to_trace[name],
                compensated_peaks,
                reduce_peak_area,
                output_path,
                name=name,
                n_radii=1
                if compensate_motion_of_peaks is None or compensate_motion is None
                else compensate_motion["n_radii"],
            )
            # numpy.save(str(traces_path), all_traces[name])

    all_smooth_traces = {}
    correlations = {}
    trace_scaling = {}
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
                        trace_scaling[recon, smooth, tgt_smooth, t] = get_trace_scaling(trace, tgt_trace)

    best_correlations = {}
    for (recon, smooth, tgt_smooth, t), corrs in correlations.items():
        best_correlations[(smooth, tgt_smooth)] = {}
        for metric, value in corrs.items():
            if metric not in best_correlations[(smooth, tgt_smooth)]:
                best_correlations[(smooth, tgt_smooth)][metric] = {}

            best = best_correlations[(smooth, tgt_smooth)][metric].get(recon, [-9999, None])[0]
            if value > best:
                best_correlations[(smooth, tgt_smooth)][metric][recon] = [float(value), t]

    print("best correlations:")
    pprint(best_correlations)
    yaml.dump(best_correlations, output_path / "best_correlations.yml")

    trace_plots_output_path = output_path / "trace_plots"
    trace_plots_output_path.mkdir(exist_ok=True)
    trace_figs = plot_traces(
        tgt=tgt,
        plots=plots,
        all_traces=all_traces,
        all_smooth_traces=all_smooth_traces,
        correlations=correlations,
        trace_scaling=trace_scaling,
        output_path=trace_plots_output_path,
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


def get_trace_scaling(trace: numpy.ndarray, tgt_trace: numpy.ndarray):
    assert len(trace.shape) == 1
    assert len(tgt_trace.shape) == 1
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(trace, tgt_trace)
    # plt.plot(trace, label="trace")
    # plt.plot(slope * trace + intercept, label="scaled")
    # plt.plot(tgt_trace, label="tgt")
    # plt.legend()
    # plt.show()
    return slope, intercept


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


def compute_compensated_peak(
    tensor,
    *,
    p,
    h,
    w,
    r,
    output_path: Path,
    method: str,
    name: str,
    n_radii: int = 3,
    accumulate_relative_motion: str = "cumsum",
    motion_decay: Optional[float] = None,
    upsample_xy: int = 1,
    upsample_t: int = 1,
    presmooth_sigma: Optional[float] = None,
):
    T, H, W = tensor.shape

    r_box = 3 * r
    # use highest fitting multiple of r to correct motion
    for nr in range(n_radii, 1, -1):
        r_box = 3 * r * nr
        if h - r_box >= 0 and h + r_box < H and w - r_box >= 0 and w + r_box < W:
            r_box = 3 * r * nr
            break
    else:
        logger.warning("peak too close to edge")
        return numpy.repeat(numpy.array([h, w, r])[:, None], T, axis=1)

    peak_vol = tensor[:, h - r_box : h + r_box + 1, w - r_box : w + r_box + 1]
    assert peak_vol.shape == (T, 2 * r_box + 1, 2 * r_box + 1)

    # volwrite(output_path / f"peak_{i}_{h}_{w}_{r}.tif", peak_vol)

    peak_motion = get_absolute_peak_motion(
        peak_vol,
        method,
        r_box // 3,
        accumulate_relative_motion=accumulate_relative_motion,
        motion_decay=motion_decay,
        upsample_xy=upsample_xy,
        upsample_t=upsample_t,
        presmooth_sigma=presmooth_sigma,
    )

    # print("\npeak_motion", peak_motion)

    intensity_at_original_peak_location = tensor[:, h, w]
    assert intensity_at_original_peak_location.shape == (T,)
    t_ref = numpy.argmax(intensity_at_original_peak_location)
    # choose reference frame to be the frame with highest intensity at the non-corrected peak location
    motion_offset = peak_motion[t_ref]
    # print("\nreference frame", t_ref, "motion offset", motion_offset)
    peak_motion -= motion_offset

    # make sure we we stay in the box
    peak_motion = numpy.clip(peak_motion, a_min=-r_box, a_max=r_box)

    assert peak_motion.shape == (T, 2), (peak_motion.shape, (T, 2))
    # print("\npeak_motion.shape", peak_motion.shape, " max peak motion", peak_motion.max())
    compensated_peak = numpy.stack([(h + dh, w + dw, r) for dh, dw in peak_motion], axis=-1)

    # write out compensated peak volume
    # compensated_peaks[p] = compensated_peak
    assert peak_vol.shape == (T, 2 * r_box + 1, 2 * r_box + 1)
    rep = 3
    peak_vol = numpy.repeat(peak_vol, rep, axis=1)
    peak_vol = numpy.repeat(peak_vol, rep, axis=2)
    peak_vol = numpy.tile(peak_vol[..., None], (1, 1, 1, 3))
    assert peak_vol.shape == (T, rep * (2 * r_box + 1), rep * (2 * r_box + 1), 3)
    center_pix = rep * r_box + rep // 2
    for t, (dh, dw) in enumerate(peak_motion):
        # ph = max(0, min(peak_vol.shape[1] - 1, center_pix + rep * dh))
        # pw = max(0, min(peak_vol.shape[2] - 1, center_pix + rep * dw))
        ph = center_pix + rep * dh
        pw = center_pix + rep * dw
        peak_vol[t, ph, pw, 0] = 0.5
        peak_vol[t, ph, pw, 1:] = 0

    peak_vol /= 5  # peak_vol.max()
    peak_vol *= 255
    peak_vol = peak_vol.clip(min=0, max=255)
    peak_vol = numpy.rint(peak_vol)
    peak_vol = peak_vol.astype("uint8")

    volwrite(output_path / f"trace_{p}_{h}_{w}_{r}_{name}_marked.tif", peak_vol)
    return compensated_peak


def get_motion_compensated_peaks(tensor: numpy.ndarray, peaks: numpy.ndarray, **kwargs):
    compensated_peaks = []
    if settings.max_workers_for_trace:
        with ProcessPoolExecutor(max_workers=settings.max_workers_for_trace) as executor:
            futs = [
                executor.submit(compute_compensated_peak, tensor, **kwargs, p=p, h=h, w=w, r=r)
                for p, (h, w, r) in enumerate(peaks)
            ]
            for fut in tqdm(as_completed(futs), total=len(futs), unit="motion_compensated_peak"):
                compensated_peaks.append(fut.result())
    else:
        for p, (h, w, r) in enumerate(peaks):
            compensated_peaks.append(compute_compensated_peak(tensor, **kwargs, p=p, h=h, w=w, r=r))

    return numpy.stack(compensated_peaks)


def get_absolute_peak_motion(
    peak_vol: numpy.ndarray,
    method: str,
    r,
    *,
    accumulate_relative_motion: str,
    motion_decay: Optional[float],
    # n_consistency_frames: int = None,
    # dt_consistency_frames: int = None,
    upsample_xy: int,
    upsample_t: int,
    presmooth_sigma: Optional[float],
):
    assert len(peak_vol.shape) == 3
    T = peak_vol.shape[0]
    assert peak_vol.shape[2] == 6 * r + 1
    assert peak_vol.shape[1] == 6 * r + 1

    assert upsample_xy >= 1
    assert upsample_t >= 1
    if upsample_xy > 1 or upsample_t > 1:
        T *= upsample_t
        out_shape = [T] + [upsample_xy * s for s in peak_vol.shape[1:]]
        peak_vol = skimage.transform.resize(peak_vol, out_shape, order=2, preserve_range=False)
        r *= upsample_xy

    if presmooth_sigma is not None:
        peak_vol = numpy.stack([gaussian_filter(pv, sigma=presmooth_sigma, mode="constant") for pv in peak_vol])

    # compute relative motion frame by frame
    if method == "home_brewed":
        roi = (2 * r, 4 * r)
        dxdy_candidates = [(i, j) for i in range(-2 * r, 2 * r + 1) for j in range(-2 * r, 2 * r + 1)]

        # def compute_dxdy(t):
        #     last_frame = peak_vol[t - 1, roi[0] : roi[1], roi[0] : roi[1]]
        #     frame_candidates = [
        #         peak_vol[t, roi[0] + dx : roi[1] + dx, roi[0] + dy : roi[1] + dy] for dx, dy in dxdy_candidates
        #     ]
        #     frame_diffs = [((last_frame - c) ** 2).sum() for c in frame_candidates]
        #     return dxdy_candidates[numpy.argmin(frame_diffs).item()]
        #
        # with ThreadPoolExecutor(max_workers=settings.max_workers_for_trace) as executor:
        #     futs = [executor.submit(compute_dxdy, t) for t in range(1, T)]
        #     rel_peak_motion = []
        #     for fut in futs:
        #         rel_peak_motion.append(fut.result())
        #
        #     rel_peak_motion = numpy.asarray(rel_peak_motion)

        rel_peak_motion = numpy.empty((T - 1, 2), dtype=numpy.int)

        def compute_dxdy(t):
            last_frame = peak_vol[t - 1, roi[0] : roi[1], roi[0] : roi[1]]
            frame_candidates = numpy.stack(
                [peak_vol[t, roi[0] + dx : roi[1] + dx, roi[0] + dy : roi[1] + dy] for dx, dy in dxdy_candidates]
            )
            frame_mse = ((frame_candidates - last_frame[None, ...]) ** 2).sum(axis=(1, 2))
            rel_peak_motion[t - 1] = dxdy_candidates[numpy.argmin(frame_mse).item()]

        # with ThreadPoolExecutor(max_workers=settings.max_workers_for_trace) as executor:
        #     [executor.submit(compute_dxdy, t) for t in range(1, T)]

        [compute_dxdy(t) for t in range(1, T)]
    else:
        block_motion = skvideo.motion.blockMotion(peak_vol, method=method, mbSize=2 * r, p=2 * r)
        rel_peak_motion = block_motion[:, block_motion.shape[1] // 2, block_motion.shape[2] // 2]
        assert rel_peak_motion.shape[1] == 2, rel_peak_motion.shape
        rel_peak_motion *= -1

    # add zero motion for first frame
    rel_peak_motion = numpy.concatenate([numpy.zeros((1, 2), dtype=numpy.int), rel_peak_motion])

    # consistency_frames = [frame]  # distant past frames for correction of absolute position
    # for t in range(1, T):
    #     if t % dt_consistency_frames == 0:
    #         consistency_frames.append(frame)

    if upsample_t > 1:
        # sum sub time step motions
        for dt in range(1, upsample_t):
            rel_peak_motion[::upsample_t] += rel_peak_motion[dt::upsample_t]

        rel_peak_motion = rel_peak_motion[::upsample_t]

    # compute absolute motion to first frame
    if accumulate_relative_motion == "cumsum":
        assert motion_decay is None
        peak_motion = numpy.cumsum(rel_peak_motion, axis=0)  # diverges (slow movement is 'rounded away'?)
    elif accumulate_relative_motion == "decaying cumsum":
        assert motion_decay is not None
        peak_motion = numpy.stack(
            [
                numpy.asarray(list(accumulate(rel_peak_motion[..., d], lambda t1, t2: (t1 * motion_decay + t2))))
                for d in range(2)
            ],
            axis=-1,
        )
    else:
        raise NotImplementedError(accumulate_relative_motion)

    peak_motion = numpy.rint(peak_motion / upsample_xy).astype(numpy.int)

    return peak_motion


@print_timing
def trace_tracked_peaks(
    tensor: numpy.ndarray, compensated_peaks: numpy.ndarray, reduce: str, output_path: Path, name: str, n_radii: int
):
    assert len(compensated_peaks.shape) == 3, compensated_peaks.shape
    P = compensated_peaks.shape[0]
    assert compensated_peaks.shape[1] == 3, compensated_peaks.shape
    R = compensated_peaks[0, -1, 0].item()
    # todo: optimize padding
    tensor = numpy.pad(
        tensor, pad_width=((0, 0), (3 * R * n_radii, 3 * R * n_radii), (3 * R * n_radii, 3 * R * n_radii)), mode="edge"
    )
    offset = numpy.zeros_like(compensated_peaks)
    offset[:, :2] = 3 * R * n_radii
    compensated_peaks = compensated_peaks + offset
    T, H, W = tensor.shape
    assert compensated_peaks.shape[2] == T, compensated_peaks.shape

    peak_vols = numpy.zeros((P, T, 2 * R, 2 * R), dtype=tensor.dtype)
    for i, compensated_peak in enumerate(compensated_peaks):
        for t, (x, y, _) in enumerate(compensated_peak.T):
            peak_vols[i, t] = tensor[t, x - R : x + R, y - R : y + R]

        # save peak volumes
        volwrite(output_path / f"trace_{i}_{name}.tif", peak_vols[i])

    peak_mask = create_circular_mask(2 * R, 2 * R, (R // 2, R // 2, R)).flatten()
    cirlce_area = peak_mask.sum()
    assert peak_mask.shape[0] == (2 * R) ** 2, (peak_mask.shape, R ** 2)

    peak_vols = peak_vols.reshape((P, T, (2 * R) ** 2))

    if reduce == "mean":
        traces = peak_vols.dot(peak_mask) / cirlce_area
    elif reduce == "max":
        traces = numpy.max(peak_vols, axis=-1, where=peak_mask[None, None, ...], initial=-9999)
        # traces = numpy.max(numpy.repeat(tensor[..., None], P, axis=-1), axis=1, where=peak_mask[None, ...])
    else:
        raise NotImplementedError(reduce)

    assert traces.shape == (P, T), (traces.shape, (P, T))
    return traces


@print_timing
def trace_straight_peaks(tensor: numpy.ndarray, peaks: numpy.ndarray, reduce: str, output_path: Path):
    assert len(peaks.shape) == 2, peaks.shape
    P = peaks.shape[0]
    assert peaks.shape[1] == 3, peaks.shape

    T, H, W = tensor.shape
    # save peak volumes
    for i, (x, y, r) in enumerate(peaks):
        r_f = 5
        peak_vol = tensor[:, x - r_f * r : x + r_f * r, y - r_f * r : y + r_f * r]

        volwrite(output_path / f"peak_{i}_{x}_{y}_{r}.tif", peak_vol)

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
        traces = numpy.max(
            numpy.repeat(tensor[..., None], P, axis=-1), axis=1, where=peak_masks[None, ...], initial=-9999
        )
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


@print_timing
def plot_traces(*, tgt, plots, all_traces, all_smooth_traces, correlations, trace_scaling, output_path, tag: str = ""):
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

    def plot_trace(ax, j, trace, name, *, i=None, wname="", w: Union[str, int] = "", w_max: int = 0, corr_key=None):
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
        if label in [ln.get_label() for ln in ax.lines]:
            # don't replot existing line
            return []

        plot_args_here = [numpy.arange(w_max // 2, all_traces[tgt].shape[1] - w_max // 2), trace]
        plot_kwargs_here = {"label": label, "color": get_color(i, j), **plot_kwargs}
        return ax.plot(*plot_args_here, **plot_kwargs_here)

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
        if all_traces[tgt].shape[0] > 1:
            # check if this plot should be skipped
            skip = True
            for recons in plots:
                for recon, kwargs in recons.items():
                    for smooth, tgt_smooth in kwargs["smooth"]:
                        if (
                            correlations[(recon, smooth, tgt_smooth, t)]["pearson"] > 0.7
                            and all_traces[tgt].max() - all_traces[tgt].min() > 0.2
                        ):
                            skip = False

            if skip:
                continue

        nrows = len(plots)
        fig, axes = plt.subplots(nrows=nrows, sharex=True, figsize=(20, 10), squeeze=False)
        axes = axes[:, 0]  # only squeeze ncols=1
        # plt.suptitle(f"Trace {i:2}")
        axes[0].set_title(f"{tag} Trace {t:2}")
        i = 0
        for ax, recons in zip(axes, plots):
            ax.tick_params(axis="y", labelcolor=get_color(None, 0))
            ax.set_xlim(0, all_traces[tgt].shape[1])
            ax.set_ylabel(trace_name_map.get(tgt, tgt), color=get_color(None, 0), fontdict=fontdict)
            plotted_lines = []
            all_twinx = {}
            tgt_min = 9999
            tgt_max = -9999
            for i, (recon, kwargs) in enumerate(recons.items()):
                for j, (smooth, tgt_smooth) in enumerate(kwargs["smooth"]):
                    twinx = ax.twinx()  # new twinx for each recon
                    # twinx = None  # plot recon on same axis as LS and scale to best fit
                    if twinx is None:
                        rel_dist_per_recon = 0
                    else:
                        all_twinx[recon, smooth, tgt_smooth, t] = twinx
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

                    traces, tgt_traces = get_smooth_traces_pair(
                        recon, smooth, tgt, tgt_smooth, all_traces, all_smooth_traces
                    )
                    slope, intercept = trace_scaling[recon, smooth, tgt_smooth, t]
                    tgt_min = min(tgt_min, slope * traces[t].min() + intercept, tgt_traces[t].min())
                    tgt_max = max(tgt_max, slope * traces[t].max() + intercept, tgt_traces[t].max())
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
                    plotted_lines += plot_trace(ax, j, tgt_traces[t], tgt, wname=tgt_wname, w=tgt_w, w_max=w_max)
                    plotted_lines += plot_trace(
                        twinx or ax,
                        j,
                        traces[t] if twinx else slope * traces[t] + intercept,
                        recon,
                        i=i,
                        wname=wname,
                        w=w,
                        w_max=w_max,
                        corr_key=(recon, smooth, tgt_smooth, t),
                    )

            labels = [l.get_label() for l in plotted_lines]

            tgt_min -= 0.01
            tgt_max += 0.01
            ax.set_ylim(tgt_min, tgt_max)
            for recon, kwargs in recons.items():
                for smooth, tgt_smooth in kwargs["smooth"]:
                    slope, intercept = trace_scaling[recon, smooth, tgt_smooth, t]
                    if numpy.isfinite([slope, intercept]).all():
                        all_twinx[recon, smooth, tgt_smooth, t].set_ylim(
                            (tgt_min - intercept) / slope, (tgt_max - intercept) / slope
                        )

            ax.legend(
                plotted_lines,
                labels,
                bbox_to_anchor=(1.0 + rel_dist_per_recon * i + 0.05, 0.5),
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

        plt.tight_layout()
        figs[f"trace{t}"] = fig
        plt.savefig(output_path / f"{t}.svg")
        plt.savefig(output_path / f"{t}.png")
        if SHOW_FIGS:
            plt.show()
        else:
            plt.close()

    return figs


def add_paths_to_plots(plots, paths):
    for plot in plots:
        for recon, kwargs in plot.items():
            kwargs["path"] = paths[recon]

    return plots
