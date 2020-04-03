from dataclasses import dataclass
from pathlib import Path


@dataclass
class DataRoots:
    GHUFNAGELLFLenseLeNet_Microscope: Path = Path("H:/")
    GKRESHUK: Path = Path("K:/")


@dataclass
class Settings:
    experiment_configs_folder: str = "experiment_configs"

    data_roots: DataRoots = DataRoots()

    num_workers_train_data_loader: int = 0
    num_workers_validate_data_loader: int = 0
    num_workers_test_data_loader: int = 0
    pin_memory: bool = False

    max_workers_per_dataset = 8
    reserved_workers_per_dataset_for_getitem = 8
    max_workers_save_output: int = 16
    max_workers_for_stat_per_ds: int = 16

    def __post_init__(self):
        assert self.reserved_workers_per_dataset_for_getitem <= self.max_workers_per_dataset

default_settings = Settings()
