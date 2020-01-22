from pathlib import Path
from typing import Tuple, Optional, List, Union, Type

import numpy

from lnet.stat import DatasetStat
from lnet.config.dataset.registration import (
    Heart_tightCrop_Transform, staticHeartFOV_Transform, wholeFOV_Transform,
    BDVTransform,
)


class PathOfInterest:
    def __init__(self, *points: Tuple[int, int, int, int], sigma: int = 1):
        self.points = points
        self.sigma = sigma


class NamedDatasetInfo:
    x_path: Path
    y_path: Path
    x_roi: Tuple[slice, slice]
    y_roi: Tuple[slice, slice, slice]
    stat: Optional[DatasetStat]
    interesting_paths: Optional[List[PathOfInterest]]

    description: str = ""
    common_path: Path = Path("/")

    def __init__(
        self,
        path: Union[str, Path],
        x_dir: str,
        y_dir: Optional[str] = None,
        description="",
        x_roi: Tuple[slice, slice] = (slice(None), slice(None)),
        y_roi: Tuple[slice, slice, slice] = (slice(None), slice(None), slice(None)),
        stat: Optional[DatasetStat] = None,
        interesting_paths: Optional[List[PathOfInterest]] = None,
        length: Optional[int] = None,
        x_shape: Optional[Tuple[int, int]] = None,
        y_shape: Optional[Tuple[int, int, int]] = None,
        AffineTransform: Optional[Union[str, Type[BDVTransform]]] = None,
    ):
        self.x_path = self.common_path / path / x_dir
        self.y_path = None if y_dir is None else self.common_path / path / y_dir
        self.description = description or self.description

        if isinstance(AffineTransform, str):
            if AffineTransform == "auto":
                posix_path = self.x_path.as_posix()
                indicators_and_AffineTransforms = {
                    "/Heart_tightCrop/": Heart_tightCrop_Transform,
                    "/staticHeartFOV/": staticHeartFOV_Transform,
                    "/wholeFOV/": wholeFOV_Transform,
                }
                for tag, TransformClass in indicators_and_AffineTransforms.items():
                    if tag in posix_path:
                        assert AffineTransform == "auto"
                        AffineTransform = TransformClass

            else:
                raise NotImplementedError(AffineTransform)

        self.AffineTransform = AffineTransform
        self.x_roi = x_roi
        self.y_roi = y_roi
        self.stat = stat
        self.interesting_paths = interesting_paths
        self.length = length

        self.x_shape = x_shape
        self.y_shape = y_shape
