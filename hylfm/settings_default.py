from dataclasses import dataclass, field
from pathlib import Path

import typing


@dataclass
class Settings:
    log_path: Path = Path(__file__).parent / "../logs"
    cache_path: Path = Path(__file__).parent / "../cache"
    experiment_configs_folder: str = "experiment_configs"

    data_roots: typing.Dict[str, Path] = field(default_factory=dict)

    num_workers_train_data_loader: int = 0
    num_workers_validate_data_loader: int = 0
    num_workers_test_data_loader: int = 0
    pin_memory: bool = False

    max_workers_per_dataset: int = 16
    reserved_workers_per_dataset_for_getitem: int = 0
    max_workers_file_logger: int = 16
    max_workers_for_hist: int = 16
    max_workers_for_stat: int = 0
    max_workers_for_trace: int = 0
    multiprocessing_start_method: str = ""

    def __post_init__(self):
        assert self.reserved_workers_per_dataset_for_getitem <= self.max_workers_per_dataset
        assert self.cache_path.exists(), self.cache_path.absolute()
