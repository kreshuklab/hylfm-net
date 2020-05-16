from __future__ import annotations

import sys
import typing
import warnings

import numpy

from lnet.transformations.affine_utils import get_crops, get_ls_shape

if typing.TYPE_CHECKING:
    from lnet.datasets import N5CachedDatasetFromInfo


def z_range(
    dataset: N5CachedDatasetFromInfo,
    idx: int,
    *,
    z_min: int = None,
    z_max: int = None,
    lf_crops: typing.Dict[str, typing.Sequence[typing.Sequence[typing.Optional[int]]]] = None,
) -> bool:
    z_slice = dataset.dataset.get_z_slice(idx)
    if z_slice is None:
        return True

    if lf_crops is None:
        if z_min is None:
            z_min = 0

        if z_max is None:
            z_max = sys.maxsize

    else:
        assert z_min is None
        assert z_max is None
        crop_name = dataset.dataset.info.meta["crop_name"]
        _, _, ls_crop = get_crops(
            crop_name, lf_crop=lf_crops[crop_name], meta=dataset.dataset.info.meta, for_slice=False
        )
        z_crop = ls_crop[1]
        z_min = z_crop[0]
        z_crop_up = z_crop[1]
        z = get_ls_shape(crop_name, for_slice=False)[0]
        if z_crop_up is None:
            z_max = z
        elif z_crop_up < 0:
            z_max = z + z_crop_up
        else:
            z_max = z_crop_up

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
