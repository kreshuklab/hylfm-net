import logging
import sys
import typing
from dataclasses import dataclass, field, make_dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


debug_mode = getattr(sys, "gettrace", None) is not None and sys.gettrace()


@dataclass
class Settings:
    log_dir: Path = Path(__file__).parent / "../logs"
    cache_dir: Path = Path(__file__).parent / "../cache"
    download_dir: Path = Path(__file__).parent / "../download"
    train_configs_dir: Path = Path(__file__).parent / "../configs/train"

    num_workers_train_data_loader: int = 0 if debug_mode else 4
    num_workers_validate_data_loader: int = 0 if debug_mode else 4
    num_workers_test_data_loader: int = 0 if debug_mode else 4
    pin_memory: bool = False

    max_workers_per_dataset: int = 0 if debug_mode else 8
    reserved_workers_per_dataset_for_getitem: int = 0
    max_workers_file_logger: int = 0 if debug_mode else 8
    max_workers_for_hist: int = 0 if debug_mode else 16
    max_workers_for_stat: int = 0 if debug_mode else 8
    max_workers_for_trace: int = 0 if debug_mode else 8
    multiprocessing_start_method: str = ""

    data_roots: typing.Dict[str, Path] = field(default_factory=dict)

    def __post_init__(self):
        assert self.reserved_workers_per_dataset_for_getitem <= self.max_workers_per_dataset
        self.log_dir = self.log_dir.absolute()
        if self.log_dir.exists():
            logger.info("logging to %s", self.log_dir.absolute())
        else:
            logger.warning("logging to %s", self.log_dir.absolute())
            self.log_dir.mkdir(parents=True, exist_ok=True)

        self.cache_dir = self.cache_dir.absolute()
        if self.cache_dir.exists():
            logger.info("caching to %s", self.cache_dir.absolute())
        else:
            logger.warning("caching to %s", self.cache_dir.absolute())
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        DataRoots = make_dataclass("DataRoots", [(name, Path) for name in self.data_roots.keys()])
        self.data_roots = DataRoots(**self.data_roots)
