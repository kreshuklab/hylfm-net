import logging
import os
import sys
import typing
import warnings
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


debug_mode = getattr(sys, "gettrace", None) is not None and sys.gettrace()


@dataclass
class Settings:
    log_dir: Path = Path(__file__).parent / "../../logs"
    download_dir: Path = Path(__file__).parent / "../../download"
    cache_dir: Path = Path(__file__).parent / "../../cache"
    configs_dir: Path = Path(__file__).parent / "../../configs"

    num_workers_train_data_loader: int = 1 if debug_mode else 2
    num_workers_validate_data_loader: int = 1 if debug_mode else 2
    num_workers_test_data_loader: int = 1 if debug_mode else 2
    pin_memory: bool = False

    max_workers_per_dataset: int = 1 if debug_mode else 4
    reserved_workers_per_dataset_for_getitem: int = 0
    max_workers_file_logger: int = 1 if debug_mode else 4
    max_workers_for_hist: int = 1 if debug_mode else 4
    max_workers_for_stat: int = 0  # 1 if debug_mode else 4
    max_workers_for_trace: int = 1 if debug_mode else 4
    multiprocessing_start_method: str = "spawn"
    OMP_NUM_THREADS: typing.Optional[int] = 1
    OPENBLAS_NUM_THREADS: typing.Optional[int] = 0
    MKL_NUM_THREADS: typing.Optional[int] = 0
    VECLIB_MAXIMUM_THREADS: typing.Optional[int] = 0
    NUMEXPR_NUM_THREADS: typing.Optional[int] = 0

    data_roots: typing.Dict[str, Path] = field(default_factory=dict)

    nice: int = 10

    def __post_init__(self):
        assert self.reserved_workers_per_dataset_for_getitem <= self.max_workers_per_dataset
        self.log_dir = Path(self.log_dir).resolve()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        logger.info("logging to %s", self.log_dir)

        self.download_dir = Path(self.download_dir).resolve()
        self.download_dir.mkdir(parents=True, exist_ok=True)
        logger.info("caching to %s", self.download_dir)

        self.cache_dir = Path(self.cache_dir).resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info("caching to %s", self.cache_dir)

        self.configs_dir = Path(self.configs_dir).resolve()
        self.configs_dir.mkdir(parents=True, exist_ok=True)
        logger.info("caching to %s", self.configs_dir)

        self.data_roots = {k: Path(v) for k, v in self.data_roots.items()}

        os.nice(self.nice)

        for numpy_env_var in [
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ]:
            value = getattr(self, numpy_env_var)
            if value is not None and not numpy_env_var in os.environ:
                if "numpy" in sys.modules:
                    warnings.warn("numpy imported before hylfm. numpy env var settings won't take effect!")

                os.environ[numpy_env_var] = str(value)

        if self.multiprocessing_start_method:
            import torch.multiprocessing
            start_method = torch.multiprocessing.get_start_method(allow_none=True)
            if start_method is None:
                torch.multiprocessing.set_start_method(self.multiprocessing_start_method)
            else:
                assert start_method == self.multiprocessing_start_method
