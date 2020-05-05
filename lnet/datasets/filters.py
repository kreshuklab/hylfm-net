from __future__ import annotations

import sys
import typing

if typing.TYPE_CHECKING:
    from lnet.datasets import N5CachedDatasetFromInfo


def z_range(dataset: N5CachedDatasetFromInfo, idx: int, z_min: int = 0, z_max: int = sys.maxsize) -> bool:
    z_slice = dataset.dataset.get_z_slice(idx)
    return z_slice is None or z_min <= z_slice < z_max


def instensity_range(
    dataset: N5CachedDatasetFromInfo,
    idx: int,
    *,
    apply_to: typing.Union[str, typing.Sequence[str]],
    min_below: typing.Optional[float] = None,
    max_above: typing.Optional[float] = None
) -> bool:
    assert min_below is not None or max_above is not None, "What's the point?"
    if isinstance(apply_to, str):
        apply_to = [apply_to]

    sample = dataset[idx]
    valid = True
    for name, tensor in sample.items():
        if name in apply_to:
            if min_below is not None:
                valid = valid and tensor.min() < min_below

            if max_above is not None:
                valid = valid and tensor.max() > max_above

    return valid


def signal_to_noise(
    dataset: N5CachedDatasetFromInfo,
    idx: int,
    *,
    apply_to: typing.Union[str, typing.Sequence[str]],
    min_below: typing.Optional[float] = None,
    max_above: typing.Optional[float] = None
) -> bool:
    assert min_below is not None or max_above is not None, "What's the point?"
    if isinstance(apply_to, str):
        apply_to = [apply_to]

    sample = dataset[idx]
    valid = True
    for name, tensor in sample.items():
        if name in apply_to:
            if min_below is not None:
                valid = valid and tensor.min() < min_below

            if max_above is not None:
                valid = valid and tensor.max() > max_above

    return valid

