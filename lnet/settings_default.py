from dataclasses import dataclass
from pathlib import Path


@dataclass
class DataRoots:
    GHUFNAGELLFLenseLeNet_Microscope: Path = Path("H:/")
    GKRESHUK: Path = Path("K:/")
    lnet: Path = Path("C:/repos/lnet")
    logs: Path = Path("K:/LF_computed/lnet/logs")

@dataclass
class Settings:
    log_path: Path
    cache_path: Path
    experiment_configs_folder: str = "experiment_configs"

    data_roots: DataRoots = DataRoots()
    wait_for_data: bool = False

    num_workers_train_data_loader: int = 0
    num_workers_validate_data_loader: int = 0
    num_workers_test_data_loader: int = 0
    pin_memory: bool = False

    max_workers_per_dataset: int = 16
    reserved_workers_per_dataset_for_getitem: int = 0
    max_workers_file_logger: int = 16
    max_workers_for_hist: int = 16

    def __post_init__(self):
        assert self.reserved_workers_per_dataset_for_getitem <= self.max_workers_per_dataset
        assert self.cache_path.exists(), self.cache_path.absolute()

