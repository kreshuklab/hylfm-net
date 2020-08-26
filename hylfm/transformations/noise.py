from typing import Optional, Tuple

import numpy

from hylfm.transformations.base import Transform


class AdditiveGaussianNoise(Transform):
    """Add gaussian noise to the input."""

    def __init__(
        self,
        sigma: Optional[float] = None,
        percentile_range: Optional[Tuple[float, float]] = None,
        scale_factor: float = 1.0,
        **super_kwargs
    ):
        super().__init__(**super_kwargs)
        if sigma is not None and percentile_range is not None:
            raise ValueError(
                "exclusive arguments: use either sigma or a percentile range (percentile_range). percentile_range is used "
                "to compute standard deviation, which will be multiplied by scale_factor (default=1.) to result in a sigma "
                "for the noise distribution. The default percentile_range is (0, 100)"
            )

        self.sigma = sigma
        self.percentile_range = percentile_range
        self.scale_factor = scale_factor

    def apply_to_sample(self, sample, tensor_name: str, tensor_idx: int, batch_idx: int, meta: dict):
        if self.sigma is None:
            mean, sigma = meta[tensor_name]["stat"].get_mean_std(
                name=tensor_name, percentile_range=self.percentile_range
            )
        else:
            sigma = self.sigma

        scale = sigma * self.scale_factor
        return sample + numpy.random.normal(loc=0, scale=scale, size=tuple(sample.shape))


class PoissonNoise(Transform):
    def __init__(
        self,
        peak: Optional[float] = None,
        peak_percentile: Optional[float] = None,
        min_peak: int = 0.001,
        seed: Optional[int] = None,
        **super_kwargs
    ):
        if peak is None and peak_percentile is None:
            raise ValueError("Require argument 'peak' or 'peak_percentile'.")

        super().__init__(**super_kwargs)
        self.generator = numpy.random.default_rng(seed=seed)
        self.peak = max(min_peak, peak)
        self.peak_percentile = peak_percentile
        self.min_peak = min_peak

    def apply_to_sample(self, sample, tensor_name: str, tensor_idx: int, batch_idx: int, meta: dict):
        if self.peak is None:
            peak = meta[tensor_name]["stat"].get_percentile(name=tensor_name, percentile=self.peak_percentile)
            print("percentile", self.peak_percentile, "peak", peak)
            peak = max(self.min_peak, peak)
        else:
            peak = self.peak

        offset = min(0, sample.min())
        return self.generator.poisson((sample - offset) * peak) / peak + offset

    # def apply_to_tensor(
    #     self, tensor: numpy.ndarray, *, name: str, idx: int, meta: List[dict]
    # ) -> numpy.ndarray:
    #     return self.generator.poisson(tensor * self.peak) / self.peak
