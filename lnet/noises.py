from typing import Generator

from inferno.io.transform import Transform
from inferno.io.transform.image import AdditiveGaussianNoise

from lnet.utils.stat import DatasetStat

def noise00(stat: DatasetStat) -> Generator[Transform, None, None]:
    yield AdditiveGaussianNoise(sigma=stat.x_std / 5, apply_to=[0])
    yield AdditiveGaussianNoise(sigma=stat.y_std, apply_to=[1])
    yield AdditiveGaussianNoise(sigma=stat.y_std / 2, apply_to=[2])

