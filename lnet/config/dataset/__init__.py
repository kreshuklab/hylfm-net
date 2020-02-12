from functools import partial
from pathlib import Path
from typing import Tuple, Optional, List, Union, Type, Sequence

import numpy

from lnet.stat import DatasetStat
from lnet.config.dataset.registration import (
    Heart_tightCrop_Transform,
    staticHeartFOV_Transform,
    wholeFOV_Transform,
    BDVTransform,
    fast_cropped_6ms_Transform,
    fast_cropped_8ms_Transform,
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
    # stat: Optional[DatasetStat]
    interesting_paths: Optional[List[PathOfInterest]]

    description: str = ""
    common_path: Path = Path("/")

    def __init__(
        self,
        path: Union[str, Path],
        x_dir: str,
        y_dir: Optional[str] = None,
        description="",
        x_roi: Optional[Tuple[slice, slice]] = None,
        y_roi: Optional[Tuple[slice, slice, slice]] = None,
        # stat: Optional[DatasetStat] = None,
        interesting_paths: Optional[List[PathOfInterest]] = None,
        length: Optional[int] = None,
        x_shape: Optional[Tuple[int, int]] = None,
        y_shape: Optional[Tuple[int, int, int]] = None,
        AffineTransform: Optional[Union[str, Type[BDVTransform]]] = None,
        z_slices: Optional[Sequence[int]] = None,
        dynamic_z_slice_mod: Optional[int] = None,
    ):
        if z_slices is not None:
            assert AffineTransform is not None

        self.x_path = self.common_path / path / x_dir
        self.y_path = None if y_dir is None else self.common_path / path / y_dir
        self.description = description or self.description

        if isinstance(AffineTransform, str):
            if AffineTransform == "from_x_path":
                posix_path = self.x_path.as_posix()
                indicators_and_AffineTransforms = {
                    "fast_cropped_6ms": fast_cropped_6ms_Transform,
                    "fast_cropped_8ms": fast_cropped_8ms_Transform,
                    "Heart_tightCrop": Heart_tightCrop_Transform,
                    "staticHeartFOV": staticHeartFOV_Transform,
                    "wholeFOV": wholeFOV_Transform,
                }
                for tag, TransformClass in indicators_and_AffineTransforms.items():
                    if tag in posix_path:
                        assert AffineTransform == "from_x_path"  # make sure tag is found only once
                        AffineTransform = TransformClass

            else:
                raise NotImplementedError(AffineTransform)

        self.DefaultAffineTransform = AffineTransform
        if AffineTransform is not None:
            x_shape = x_shape or AffineTransform.lf_shape[1:]

            auto_y_shape = tuple(
                y - y_crop[0] - y_crop[1] for y_crop, y in zip(AffineTransform.lf2ls_crop, AffineTransform.ls_shape)
            )
            auto_y_roi = tuple(
                slice(y_crop[0], y - y_crop[1])
                for y_crop, y in zip(AffineTransform.lf2ls_crop, AffineTransform.ls_shape)
            )
            if z_slices is not None or dynamic_z_slice_mod is not None:
                # auto_y_shape = auto_y_shape[1:]
                auto_y_roi = auto_y_roi[1:]

            y_shape = y_shape or auto_y_shape
            y_roi = y_roi or auto_y_roi

            if z_slices is not None or dynamic_z_slice_mod is not None:
                assert len(y_shape) == 3
                assert len(y_roi) == 2

        self.x_roi = x_roi or (slice(None), slice(None))
        self.y_roi = y_roi or (slice(None), slice(None), slice(None))
        self.interesting_paths = interesting_paths
        self.length = length

        self.x_shape = x_shape
        self.y_shape = y_shape

        self.z_slices = z_slices
        self.dynamic_z_slice_mod = dynamic_z_slice_mod
