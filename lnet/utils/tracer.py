import json
import logging
import math
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
from hashlib import sha256
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import matplotlib.pyplot as plt
import numpy
import scipy.signal
import skimage.transform
import skvideo.io
import skvideo.motion
import tifffile
from imageio import imread, imwrite
from ruamel.yaml import YAML
from scipy.ndimage import gaussian_filter
from scipy.stats import pearsonr, spearmanr
from skimage.feature import peak_local_max
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from lnet import settings
from lnet.datasets import TensorInfo
from lnet.datasets.base import TiffDataset, get_collate_fn
from lnet.utils.general import print_timing

logger = logging.getLogger(__name__)

yaml = YAML(typ="safe")

SHOW_FIGS = False

# use tifffile instead of imageio, because imageio.volwrite(p, data) throws an exception when passing 'compress' kwarg
def volwrite(p: Path, data, compress=2, **kwargs):
    with p.open("wb") as f:
        tifffile.imwrite(f, data, compress=compress, **kwargs)


@print_timing
def trace(
    tgt_path: Union[str, Path],
    tgt: str,
    roi: Tuple[slice, slice],
    plots: List[Union[Dict[str, Dict[str, Union[str, Path, List]]], Set[str]]],
    output_path: Path,
    nr_traces: int,
    background_threshold: float,
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
    if isinstance(tgt_path, str):
        tgt_path = Path(tgt_path)

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
    assert length <= 600, "across TP??"
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

    figs = {}
    peak_path = output_path / f"{tgt}_peaks_of_{compute_peaks_on}.yml"
    peaks = None
    if peak_path.exists() and not overwrite_existing_files:
        peaks = numpy.asarray(yaml.load(peak_path))

        if peaks.shape[0] != nr_traces:
            peaks = None

    if peaks is None or plot_peaks:
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
            background_mask = (all_projections[tgt]["min"] > background_threshold).astype(numpy.int)
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

            fig_name = f"trace_positions_on_{tensor_name}"
            plt.savefig(output_path / f"{fig_name}.svg")
            plt.savefig(output_path / f"{fig_name}.png")
            figs[fig_name] = fig
            if SHOW_FIGS:
                plt.show()
            else:
                plt.close()

            yaml.dump(peaks.tolist(), peak_path)

    if compensate_motion_of_peaks:
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
                datasets_to_trace[name], compensated_peaks, reduce_peak_area, output_path, name=name
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
        for metric, value in corrs.items():
            if metric not in best_correlations:
                best_correlations[metric] = {}

            best = best_correlations[metric].get(recon, [-9999, None])[0]
            if value > best:
                best_correlations[metric][recon] = [float(value), t]

    print("best correlations:", best_correlations)
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
):
    T, H, W = tensor.shape

    r *= n_radii
    if h - 3 * r < 0 or h + 3 * r >= H or w - 3 * r < 0 or w + 3 * r >= W:
        raise NotImplementedError("peak too close to edge")

    peak_vol = tensor[:, h - 3 * r : h + 3 * r, w - 3 * r : w + 3 * r]

    # volwrite(output_path / f"peak_{i}_{h}_{w}_{r}.tif", peak_vol)

    peak_motion = get_absolute_peak_motion(
        peak_vol,
        method,
        r,
        accumulate_relative_motion=accumulate_relative_motion,
        motion_decay=motion_decay,
        upsample_xy=upsample_xy,
    )

    # print("\npeak_motion", peak_motion)

    intensity_at_original_peak_location = tensor[:, h, w]
    assert intensity_at_original_peak_location.shape == (T,)
    t_ref = numpy.argmax(intensity_at_original_peak_location)
    # choose reference frame to be the frame with highest intensity at the non-corrected peak location
    motion_offset = peak_motion[t_ref]
    # print("\nreference frame", t_ref, "motion offset", motion_offset)
    peak_motion -= motion_offset

    assert peak_motion.shape == (T, 2), (peak_motion.shape, (T, 2))
    # print("\npeak_motion.shape", peak_motion.shape, " max peak motion", peak_motion.max())
    compensated_peak = numpy.stack([(h + dh, w + dw, r) for dh, dw in peak_motion], axis=-1)
    # compensated_peaks[p] = compensated_peak

    # write out compensated peak volume
    rep = 3
    peak_vol = numpy.repeat(peak_vol, rep, axis=1)
    peak_vol = numpy.repeat(peak_vol, rep, axis=2)
    peak_vol = numpy.tile(peak_vol[..., None], (1, 1, 1, 3))
    center_pix = rep // 2 + rep * 3 * r
    for t, (dh, dw) in enumerate(peak_motion):
        peak_vol[t, center_pix + rep * dh, center_pix + rep * dw, 0] = 0.5
        peak_vol[t, center_pix + rep * dh, center_pix + rep * dw, 1:] = 0

    peak_vol -= peak_vol.min()
    peak_vol /= peak_vol.max()
    peak_vol *= 255
    peak_vol = peak_vol.astype("uint8")
    volwrite(output_path / f"peak_{p}_{h}_{w}_{r}_{name}_marked.tif", peak_vol)
    return compensated_peak


def get_motion_compensated_peaks(tensor: numpy.ndarray, peaks: numpy.ndarray, **kwargs):
    compensated_peaks = []
    with ProcessPoolExecutor(max_workers=settings.max_workers_for_trace) as executor:
        futs = [
            executor.submit(compute_compensated_peak, tensor, **kwargs, p=p, h=h, w=w, r=r)
            for p, (h, w, r) in enumerate(peaks)
        ]
        for fut in tqdm(as_completed(futs), total=len(futs), unit="motion_compensated_peak"):
            try:
                comp_peak = fut.result()
            except Exception as e:
                logger.error(e)
            else:
                compensated_peaks.append(comp_peak)

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
):
    assert len(peak_vol.shape) == 3
    T = peak_vol.shape[0]
    assert peak_vol.shape[1] == 6 * r
    assert peak_vol.shape[2] == 6 * r

    assert upsample_xy >= 1
    if upsample_xy > 1:
        out_shape = [peak_vol.shape[0]] + [upsample_xy * s for s in peak_vol.shape[1:]]
        peak_vol = skimage.transform.resize(peak_vol, out_shape, order=2, preserve_range=False)
        r *= upsample_xy

    # compute relative motion frame by frame
    if method == "home_brewed":
        roi = (2 * r, 4 * r)
        dxdy_candidates = [numpy.array([i, j]) for i in range(-2 * r, 2 * r + 1) for j in range(-2 * r, 2 * r + 1)]

        rel_peak_motion = numpy.empty((T - 1, 2), dtype=numpy.int)

        def compute_dxdy(t):
            last_frame = peak_vol[t - 1, roi[0] : roi[1], roi[0] : roi[1]]
            frame_candidates = [
                peak_vol[t, roi[0] + dx : roi[1] + dx, roi[0] + dy : roi[1] + dy] for dx, dy in dxdy_candidates
            ]
            frame_diffs = [((last_frame - c) ** 2).sum() for c in frame_candidates]
            rel_peak_motion[t - 1] = dxdy_candidates[numpy.argmin(frame_diffs).item()]

        # with ThreadPoolExecutor(max_workers=settings.max_workers_for_trace) as executor:
        #     [executor.submit(compute_dxdy, t) for t in range(1, T)]

        [compute_dxdy(t) for t in range(1, T)]
    else:
        block_motion = skvideo.motion.blockMotion(peak_vol, method=method, mbSize=2 * r, p=2 * r)
        rel_peak_motion = block_motion[:, block_motion.shape[1] // 2, block_motion.shape[2] // 2]
        assert rel_peak_motion.shape[1] == 2, rel_peak_motion.shape
        rel_peak_motion *= -1

    # consistency_frames = [frame]  # distant past frames for correction of absolute position
    # for t in range(1, T):
    #     if t % dt_consistency_frames == 0:
    #         consistency_frames.append(frame)

    # compute absolute motion to first frame
    if accumulate_relative_motion == "cumsum":
        assert motion_decay is None
        peak_motion = numpy.cumsum(rel_peak_motion, axis=0)  # diverges (slow movement is 'rounded away'?)
    elif accumulate_relative_motion == "decaying cumsum":
        assert motion_decay is not None
        peak_motion = scipy.signal.lfilter([motion_decay], [1, -motion_decay], rel_peak_motion, axis=-1)
        # scipy equivalent for scipy.signal.lfilter:
        # peak_motion = numpy.stack(
        #     [
        #         numpy.asarray(list(accumulate(rel_peak_motion[..., d], lambda x, y: (x + y) * motion_decay)))
        #         for d in range(2)
        #     ],
        #     axis=-1,
        # )
    else:
        raise NotImplementedError(accumulate_relative_motion)

    peak_motion = numpy.rint(peak_motion / upsample_xy).astype(numpy.int)

    # add zero motion for first frame
    peak_motion = numpy.concatenate([numpy.zeros((1, 2), dtype=numpy.int), peak_motion])
    return peak_motion


@print_timing
def trace_tracked_peaks(
    tensor: numpy.ndarray, compensated_peaks: numpy.ndarray, reduce: str, output_path: Path, name: str
):
    T, H, W = tensor.shape
    assert compensated_peaks.shape[1] == 3, compensated_peaks.shape
    assert len(compensated_peaks.shape) == 3, compensated_peaks.shape
    P = compensated_peaks.shape[0]
    assert compensated_peaks.shape[2] == T, compensated_peaks.shape

    R = compensated_peaks[0, -1, 0]
    peak_vols = numpy.zeros((P, T, 2 * R, 2 * R), dtype=tensor.dtype)
    for i, compensated_peak in enumerate(compensated_peaks):
        for t, (x, y, _) in enumerate(compensated_peak.T):
            peak_vols[i, t] = tensor[t, x - R : x + R, y - R : y + R]

        # save peak volumes
        volwrite(output_path / f"peak_{i}_{name}.tif", peak_vols[i])

    peak_mask = create_circular_mask(2 * R, 2 * R, (R // 2, R // 2, R)).flatten()
    cirlce_area = peak_mask.sum()
    assert peak_mask.shape[0] == (2 * R) ** 2, (peak_mask.shape, R ** 2)

    peak_vols = peak_vols.reshape((P, T, (2 * R) ** 2))

    if reduce == "mean":
        traces = peak_vols.dot(peak_mask) / cirlce_area
    elif reduce == "max":
        traces = numpy.max(peak_vols, axis=1, where=peak_mask[None, None, ...])
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
    paths_for_tags = {}
    rois = {}

    for tag in ["09_3__2020-03-09_06.43.40__SinglePlane_-330"]:
        paths_for_tags[tag] = {
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
        paths_for_tags[tag] = {
            name: Path(
                f"/g/kreshuk/LF_computed/lnet/logs/brain1/test_z_out49/lr_f4/20-06-12_22-07-43/brain.{tag}/run000/ds0-0"
            )
            for name in ["ls_slice", "lr_slice"]
        }
        paths_for_tags[tag]["pred"] = Path(
            f"/g/kreshuk/LF_computed/lnet/logs/brain1/test_z_out49/f4/z_out49/f4_b2_only11_2/20-06-06_17-59-42/v1_checkpoint_29500_MS_SSIM=0.8786535175641378/brain.{tag}/run000/ds0-0"
        )
        rois[tag] = (slice(25, 225), slice(55, 305))

    for i, (tag, time_range) in enumerate(
        [
            ("09_3__2020-03-09_06.43.40__SinglePlane_-330", (10, 600)),
            ("09_3__2020-03-09_06.43.40__SinglePlane_-330", (610, None)),
            ("11_2__2020-03-11_06.53.14__SinglePlane_-330", (10, None)),
            ("11_2__2020-03-11_07.30.39__SinglePlane_-310", (10, None)),
            ("11_2__2020-03-11_07.30.39__SinglePlane_-320", (10, None)),
            ("11_2__2020-03-11_10.13.20__SinglePlane_-290", (10, None)),
            ("11_2__2020-03-11_10.17.34__SinglePlane_-280", (10, None)),
            ("11_2__2020-03-11_10.17.34__SinglePlane_-330", (10, None)),
            ("11_2__2020-03-11_10.21.14__SinglePlane_-295", (10, None)),
            ("11_2__2020-03-11_10.21.14__SinglePlane_-305", (10, None)),
            ("11_2__2020-03-11_10.25.41__SinglePlane_-295", (10, None)),
            ("11_2__2020-03-11_10.25.41__SinglePlane_-340", (10, None)),
        ]
    ):
        paths = paths_for_tags[tag]
        # paths = paths_09_3_a[330]
        output_path = Path(f"/g/kreshuk/LF_computed/lnet/traces/{tag}")
        tgt = "ls_slice"
        plots = add_paths_to_plots(
            [
                {"lr_slice": {"smooth": [(None, None)]}, "pred": {"smooth": [(None, None)]}},
                {
                    "lr_slice": {"smooth": [(("flat", 11), ("flat", 11))]},
                    "pred": {"smooth": [(("flat", 11), ("flat", 11))]},
                },
                # {"lr_slice": {"smooth": [(None, ("flat", 3))]}, "pred": {"smooth": [(None, ("flat", 3))]}},
                # {"lr_slice": {"smooth": [(None, None)]}},
                # {"lr_slice": {"smooth": [(("flat", 11), ("flat", 11))]}},
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
                    "pred": {
                        "smooth": [
                            (
                                ("savgol_filter", {"window_length": 11, "polyorder": 3}),
                                ("savgol_filter", {"window_length": 11, "polyorder": 3}),
                            )
                        ]
                    },
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
        try:
            peaks, traces, correlations, figs, motion = trace(
                tgt_path=paths[tgt],
                tgt=tgt,
                roi=rois.get(tag, (slice(0, 9999), slice(0, 9999))),
                plots=plots,
                output_path=output_path,
                nr_traces=64,
                background_threshold=0.1,
                overwrite_existing_files=False,
                smooth_diff_sigma=1.3,
                peak_threshold_abs=0.05,
                reduce_peak_area="mean",
                plot_peaks=False,
                compute_peaks_on="max",  # std, diff, min, max, mean
                peaks_min_dist=3,
                trace_radius=2,
                # time_range=(0, 600),
                # time_range=(0, 50),
                # time_range=(660, 1200),
                time_range=time_range,
                # compensate_motion={"compensate_ref": tgt, "method": "ES", "mbSize": 50, "p": 4},
                compensate_motion={
                    "of_peaks": True,
                    "only_on_tgt": True,
                    "method": "home_brewed",
                    "n_radii": 4,
                    "accumulate_relative_motion": "decaying cumsum",
                    "motion_decay": 0.8,
                    "upsample_xy": 2,
                },
                # "ES" --> exhaustive search
                # "3SS" --> 3-step search
                # "N3SS" --> "new" 3-step search [#f1]_
                # "SE3SS" --> Simple and Efficient 3SS [#f2]_
                # "4SS" --> 4-step search [#f3]_
                # "ARPS" --> Adaptive Rood Pattern search [#f4]_
                # "DS" --> Diamond search [#f5]_
                tag=tag,
            )
        except Exception as e:
            logger.error(e, exc_info=True)
