from typing import Generator, Optional, Tuple, Callable, Union, List

from inferno.io.transform.image import AdditiveGaussianNoise

from lnet.stat import DatasetStat


def additive_gaussian_noise(
    apply_to: Union[int, List[int]],
    sigma: Optional[float] = None,
    percentile_range: Optional[Tuple[float, float]] = None,
    std_factor: float = 1.0,
) -> Callable[[DatasetStat], Generator[DatasetStat, None, None]]:
    if isinstance(apply_to, int):
        apply_to = [apply_to]

    if sigma is not None and percentile_range is not None:
        raise ValueError(
            "exclusive arguments: use either sigma or a percentile range (percentile_range). percentile_range is used "
            "to compute standard deviation, which will be multiplied by std_factor (default=1.) to result in a sigma "
            "for the noise distribution. The default percentile_range is (0, 100)"
        )

    def gaussian_noise_impl(stat: DatasetStat) -> Generator[DatasetStat, None, None]:
        for idx in apply_to:
            if sigma is None:
                mean, std = stat.get_mean_std(idx=idx, percentile_range=percentile_range)
                sigma = std * std_factor

            yield AdditiveGaussianNoise(sigma=sigma, apply_to=[idx])

    return gaussian_noise_impl
