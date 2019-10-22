from typing import Generator, Optional, Tuple, Callable, List, Union

from inferno.io.transform.generic import Normalize

from lnet.utils.transforms import Clip, Normalize01Sig, Normalize01
from lnet.stat import DatasetStat


def norm(
    apply_to: Union[int, List[int]],
    mean: Optional[float] = None,
    std: Optional[float] = None,
    percentile_range: Optional[Tuple[Optional[float], Optional[float]]] = None,
) -> Callable[[DatasetStat], Generator[DatasetStat, None, None]]:
    if isinstance(apply_to, int):
        apply_to = [apply_to]

    if mean is None and std is not None:
        raise ValueError("standard deviation (std) is specified, but mean is not!")

    if mean is not None and std is None:
        raise ValueError("mean is specified, but standard deviation (std) is not!")

    if mean is not None and std is not None and percentile_range is not None:
        raise ValueError(
            "exclusive arguments: use either mean and standard deviation (std) or a percentile range "
            "(percentile_range) to compute these. Default is percentile_range=(0, 100)"
        )

    if all(arg is None for arg in [mean, std, percentile_range]):
        percentile_range = (0.0, 100.0)

    def norm_impl(stat: DatasetStat) -> Generator[DatasetStat, None, None]:
        if mean is None:
            for idx in apply_to:
                mean, std = stat.get_mean_std(idx=idx, percentile_range=percentile_range)
                yield Normalize(mean=mean, std=std, apply_to=[idx])
        else:
            yield Normalize(mean=mean, std=std, apply_to=apply_to)

    return norm_impl


def norm01(
    apply_to: Union[int, List[int]],
    min_: Optional[float] = None,
    max_: Optional[float] = None,
    percentile_min: Optional[float] = None,
    percentile_max: Optional[float] = None,
) -> Callable[[DatasetStat], Generator[DatasetStat, None, None]]:
    if isinstance(apply_to, int):
        apply_to = [apply_to]

    for name, limit, plimit in [["min", min_, percentile_min], ["max", max_, percentile_max]]:
        if limit is not None and plimit is not None:
            raise ValueError(f"exclusive arguments: use either {name} or percentile_{name}")

    if min_ is None and percentile_min is None:
        percentile_min = 0.0

    if max_ is None and percentile_max is None:
        percentile_max = 1000.0

    def norm_impl(stat: DatasetStat) -> Generator[DatasetStat, None, None]:
        req = []
        if min_ is None:
            req.append(percentile_min)

        if max_ is None:
            req.append(percentile_max)

        for idx in apply_to:
            stat.request(percentiles={idx: set(req)})

            min_max = stat.get_percentiles(idx=idx, percentiles=req)

            if max_ is None:
                max_ = min_max.pop()

            if min_ is None:
                min_ = min_max.pop()

            yield Normalize01(min_=min_, max_=max_, clip=False, apply_to=[idx])

    return norm_impl
