from pathlib import Path
from typing import Tuple, Optional, List, Union

from lnet.stat import DatasetStat


class PathOfInterest:
    def __init__(self, *points: Tuple[int, int, int, int], sigma: int = 1):
        self.points = points
        self.sigma = sigma


class DatasetConfigEntry:
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
        y_dir: str,
        description="",
        x_roi: Tuple[slice, slice] = (slice(None), slice(None)),
        y_roi: Tuple[slice, slice, slice] = (slice(None), slice(None), slice(None)),
        stat: Optional[DatasetStat] = None,
        interesting_paths: Optional[List[PathOfInterest]] = None,
    ):
        self.x_path = self.common_path / path / x_dir
        self.y_path = self.common_path / path / y_dir
        self.description = description or self.description

        self.x_roi = x_roi
        self.y_roi = y_roi
        self.stat = stat
        self.interesting_paths = interesting_paths
