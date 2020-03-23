from typing import Dict, Optional, Tuple

import numpy

from lnet.transforms.base import Transform


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

    def apply_to_sample(self, sample, tensor_name: str, tensor_idx: int, batch_idx: int, meta: Optional[Dict]):
        if self.sigma is None:
            mean, sigma = meta["stat"].get_mean_std(idx=tensor_idx, percentile_range=self.percentile_range)
        else:
            sigma = self.sigma

        scale = sigma * self.scale_factor
        return sample + numpy.random.normal(loc=0, scale=scale, size=tuple(sample.shape))
