from typing import Any, Dict, Optional, Tuple

import numpy
import skimage.util

from hylfm.hylfm_types import Array
from hylfm.stat_ import DatasetStat
from .base import Transform

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


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
            mean, sigma = stat.get_mean_std(name=self.apply_to, percentile_range=self.percentile_range_to_compute_sigma)
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


class RandomNoise(Transform):
    def __init__(
        self,
        *,
        mode: Literal["gaussian", "localvar", "poisson", "salt", "pepper", "s&p", "speckle"],
        seed: Optional[int] = None,
        clip: bool,
        mean: Optional[float],
        var: Optional[float],
        local_vars: Optional[numpy.ndarray],
        amount: Optional[float],
        salt_vs_pepper: Optional[float],
        **super_kwargs,
    ):
        self.mode = mode
        self.seed = seed
        self.clip = clip
        self.mean = mean
        self.var = var
        self.local_vars = local_vars
        self.amount = amount
        self.salt_vs_pepper = salt_vs_pepper
        super().__init__(**super_kwargs)

    def apply_to_batch(self, image: Array) -> Dict[str, Any]:
        return skimage.util.random_noise(
            image,
            mode=self.mode,
            seed=self.seed,
            clip=self.clip,
            mean=self.mean,
            var=self.var,
            local_vars=self.local_vars,
            amount=self.amount,
            salt_vs_pepper=self.salt_vs_pepper,
        )

    apply_to_sample = apply_to_batch
