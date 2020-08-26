import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


debug_mode = getattr(sys, "gettrace", None) is not None and sys.gettrace()


@dataclass
class Settings:
    log_path: Path = Path(__file__).parent / "../logs"
    cache_path: Path = Path(__file__).parent / "../cache"
    download_path: Path = Path(__file__).parent / "../download"

    experiment_configs_folder: str = "experiment_configs"

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

    def __post_init__(self):
        assert self.reserved_workers_per_dataset_for_getitem <= self.max_workers_per_dataset
        self.log_path = self.log_path.absolute()
        if self.log_path.exists():
            logger.info("logging to %s", self.log_path.absolute())
        else:
            logger.warning("logging to %s", self.log_path.absolute())
            self.log_path.mkdir(parents=True, exist_ok=True)

        self.cache_path = self.cache_path.absolute()
        if self.cache_path.exists():
            logger.info("caching to %s", self.cache_path.absolute())
        else:
            logger.warning("caching to %s", self.cache_path.absolute())
            self.cache_path.mkdir(parents=True, exist_ok=True)
