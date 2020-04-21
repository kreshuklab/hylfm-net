from __future__ import annotations

import sys
import typing

if typing.TYPE_CHECKING:
    from lnet.datasets import N5CachedDatasetFromInfo


def z_range(dataset: N5CachedDatasetFromInfo, idx: int, z_min: int = 0, z_max: int = sys.maxsize) -> bool:
    return z_min <= dataset.dataset.get_z_slice(idx) < z_max
