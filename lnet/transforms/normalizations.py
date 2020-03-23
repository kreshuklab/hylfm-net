from typing import Any, Dict, Optional, Tuple

import numpy

from lnet.transforms.base import Transform


class Normalize01(Transform):
    def __init__(
        self,
        min_: Optional[float] = None,
        max_: Optional[float] = None,
        min_percentile: Optional[float] = None,
        max_percentile: Optional[float] = None,
        clip: bool = False,
        **super_kwargs,
    ):
        super().__init__(**super_kwargs)
        if min_ is not None and min_percentile is not None:
            raise ValueError(f"exclusive arguments: use either `min_` or `min_percentile`")

        if max_ is not None and max_percentile is not None:
            raise ValueError(f"exclusive arguments: use either `max_` or `max_percentile`")

        if min_ is None and min_percentile is None:
            min_percentile = 0.0

        if max_ is None and max_percentile is None:
            max_percentile = 100.0

        self.min = min_
        self.min_percentile = min_percentile
        self.max = max_
        self.max_percentile = max_percentile
        self.clip = clip

    def apply_to_sample(self, sample, tensor_name: str, tensor_idx: int, batch_idx: int, meta: Optional[Dict]):
        percentiles2compute = [
            p for p, m in [(self.min_percentile, self.min), (self.max_percentile, self.max)] if m is None
        ]
        if percentiles2compute:
            min_max = meta["stat"].get_percentiles(idx=tensor_idx, percentiles=percentiles2compute)
        else:
            min_max = []

        max_ = self.max or min_max.pop()
        min_ = self.min or min_max.pop()
        assert not min_max, min_max
        assert min_ < max_
        sample = (sample - min_) / (max_ - min_)
        if self.clip:
            sample = numpy.clip(sample, 0.0, 1.0)

        return sample


class NormalizeMeanStd(Transform):
    def __init__(
        self,
        mean: Optional[float] = None,
        std: Optional[float] = None,
        percentile_min: Optional[float] = None,
        percentile_max: Optional[float] = None,
        percentile_range: Optional[Tuple[float, float]] = None,
        epsilon: float = 1e-4,
        **super_kwargs,
    ):
        super().__init__(**super_kwargs)
        if mean is None and std is not None:
            raise ValueError("standard deviation `std` is specified, but `mean` is not!")

        if mean is not None and std is None:
            raise ValueError("mean is specified, but standard deviation `std` is not!")

        if mean is not None and std is not None and percentile_range is not None:
            raise ValueError(
                "exclusive arguments: use either mean and standard deviation `std` or a percentile range "
                "`percentile_range` to compute these. Default is `percentile_range`=(0, 100)"
            )

        if percentile_range is not None and (percentile_min is not None or percentile_max is not None):
            raise ValueError("exclusive arguments: `percentile_range` and (`percentile_min`, `percentile_max`)")

        if percentile_range is None and (percentile_min is not None or percentile_max is not None):
            if percentile_min is None:
                percentile_min = 0

            if percentile_max is None:
                percentile_max = 100.0

            percentile_range = (percentile_min, percentile_max)

        if all(arg is None for arg in [mean, std, percentile_range]):
            percentile_range = (0.0, 100.0)

        self.percentile_range = percentile_range
        self.mean = mean
        self.std = std
        self.epsilon = epsilon

    def apply_to_sample(
        self, sample, tensor_name: str, tensor_idx: int, batch_idx: int, meta: Optional[Dict[str, Any]]
    ):
        if self.mean is None:
            assert self.std is None
            mean, std = meta["stat"].get_mean_std(idx=tensor_idx, percentile_range=self.percentile_range)
        else:
            assert self.std is not None
            mean, std = self.mean, self.std

        return (sample - mean) / (std + self.epsilon)
