from typing import Generator

from inferno.io.transform import Transform
from inferno.io.transform.generic import Normalize

from lnet.utils.transforms import (
    Clip,
    Normalize01Sig,
    Normalize01,
)
from lnet.stat import DatasetStat


def norm00(stat: DatasetStat) -> Generator[Transform, None, None]:
    yield Normalize(mean=stat.x_mean, std=stat.x_std, apply_to=[0])
    yield Normalize01Sig(min_=stat.corr_y_min, max_=stat.corr_y_max, apply_to=[1])
    yield Normalize01Sig(min_=stat.corr_y_min, max_=stat.corr_y_max / 2, apply_to=[2])


def norm01(stat: DatasetStat) -> Generator[Transform, None, None]:
    yield Normalize(mean=stat.x_mean, std=stat.x_std, apply_to=[0])
    yield Normalize01Sig(min_=stat.corr_y_min, max_=stat.corr_y_max, apply_to=[1])
    yield Normalize01Sig(min_=stat.corr_y_min, max_=stat.corr_y_max, apply_to=[2])
