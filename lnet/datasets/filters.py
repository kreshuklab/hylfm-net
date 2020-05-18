from __future__ import annotations

import sys
import typing
import warnings

import numpy

from lnet.transformations.affine_utils import get_ls_roi

if typing.TYPE_CHECKING:
    from lnet.datasets import N5CachedDatasetFromInfo


def z_range(dataset: N5CachedDatasetFromInfo, idx: int, *, z_min: int = None, z_max: int = None) -> bool:
    z_slice = dataset.dataset.get_z_slice(idx)
    if z_slice is None:
        return True

    if z_min is None:
        assert z_max is None
        crop_name = dataset.dataset.info.meta["crop_name"]
        meta = dataset.dataset.info.meta
        ls_roi = get_ls_roi(
            crop_name,
            pred_z_min=meta["pred_z_min"],
            pred_z_max=meta["pred_z_max"],
            for_slice=False,
            shrink=meta["shrink"],
            scale=meta["scale"],
            nnum=meta["nnum"],
            wrt_ref=True,
            z_ls_rescaled=meta["z_ls_rescaled"],
            ls_scale=meta.get("ls_scale", meta["scale"]),
        )
        z_min, z_max = ls_roi[0]
    else:
        assert z_max is not None, "z_max is missing"

    return z_min <= z_slice < z_max


def instensity_range(
    dataset: N5CachedDatasetFromInfo,
    idx: int,
    *,
    apply_to: typing.Union[str, typing.Sequence[str]],
    min_below: typing.Optional[float] = None,
    max_above: typing.Optional[float] = None,
) -> bool:
    assert min_below is not None or max_above is not None, "What's the point?"
    if isinstance(apply_to, str):
        apply_to = [apply_to]

    sample = dataset[idx]
    for name, tensor in sample.items():
        if name in apply_to:
            if min_below is not None and tensor.min() > min_below:
                return False

            if max_above is not None and tensor.max() < max_above:
                return False

    return True


def signal2noise(
    dataset: N5CachedDatasetFromInfo,
    idx: int,
    *,
    apply_to: typing.Union[str, typing.Sequence[str]],
    signal_percentile: float,
    noise_percentile: float,
    ratio: float,
) -> bool:
    if isinstance(apply_to, str):
        apply_to = [apply_to]

    sample = dataset[idx]
    for name, tensor in sample.items():
        if name in apply_to:
            signal = numpy.percentile(tensor, signal_percentile)
            noise = numpy.percentile(tensor, noise_percentile)
            if not noise:
                warnings.warn(f"encountered noise=0, signal={signal}")
                return False

            if signal / noise < ratio:
                return False

    return True
