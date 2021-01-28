from typing import Optional, Tuple

import numpy

from .base import Transform
from ..hylfm_types import Array
from ..stat_ import DatasetStat


class AdditiveGaussianNoise(Transform):
    """Add gaussian noise to the input."""

    def __init__(
        self,
        *,
        sigma: Optional[float] = None,
        percentile_range_to_compute_sigma: Tuple[float, float] = (0.0, 100.0),
        scale_factor: float = 1.0,
        apply_to: str,
    ):
        assert isinstance(apply_to, str)
        super().__init__(input_mapping={apply_to: "tensor", "stat": "stat"}, output_mapping={"tensor": apply_to})
        self.apply_to = apply_to
        if sigma is not None and percentile_range_to_compute_sigma != (0.0, 100.0):
            raise ValueError("exclusive arguments: use either sigma or percentile_range_to_compute_sigma.")

        self.sigma = sigma
        self.percentile_range_to_compute_sigma = percentile_range_to_compute_sigma
        self.scale_factor = scale_factor

    def apply_to_sample(self, tensor, stat: DatasetStat):
        if not isinstance(tensor, numpy.ndarray):
            raise NotImplementedError(type(tensor))

        assert isinstance(stat, DatasetStat)
        if self.sigma is None:
            mean, sigma = stat[self.apply_to].get_mean_std(name=self.apply_to, percentile_range=self.percentile_range_to_compute_sigma)
        else:
            sigma = self.sigma

        scale = sigma * self.scale_factor
        if numpy.issubdtype(tensor.dtype, numpy.floating):
            tensor += numpy.random.normal(loc=0, scale=scale, size=tuple(tensor.shape))
            return tensor
        else:
            return tensor + numpy.random.normal(loc=0, scale=scale, size=tuple(tensor.shape))


class PoissonNoise(Transform):
    def __init__(
        self,
        *,
        peak: Optional[float] = None,
        peak_percentile: Optional[float] = None,
        min_peak: float = 0.001,
        seed: Optional[int] = None,
        apply_to: str,
    ):
        assert isinstance(apply_to, str)
        super().__init__(input_mapping={apply_to: "tensor", "stat": "stat"}, output_mapping={"tensor": apply_to})
        self.apply_to = apply_to
        if peak is None and peak_percentile is None:
            raise ValueError("Require argument 'peak' or 'peak_percentile'.")

        self.generator = numpy.random.default_rng(seed=seed)
        self.peak = max(min_peak, peak)
        self.peak_percentile = peak_percentile
        self.min_peak = min_peak

    def apply_to_sample(self, tensor: Array, stat: DatasetStat):
        if self.peak is None:
            peak = stat[self.apply_to].get_percentile(name=self.apply_to, percentile=self.peak_percentile)
            peak = max(self.min_peak, peak)
        else:
            peak = self.peak

        offset = min(0, tensor.min())
        return self.generator.poisson((tensor - offset) * peak) / peak + offset
