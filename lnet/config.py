import logging
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class ConfigBase:
    def __post_init__(self):
        # initialize nested dataclasses
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, dict):
                try:
                    if is_dataclass(f.type):
                        dclass = f.type
                    elif (
                        hasattr(f.type, "__origin__")
                        and f.type.__origin__ is Union
                        and is_dataclass(f.type.__args__[0])
                        and f.type.__args__[1].__name__ == "NoneType"
                    ):  # optional dataclass
                        dclass = f.type.__args__[0]
                    elif hasattr(f.type, "__origin__") and issubclass(f.type.__origin__, dict):  # typing.Dict
                        continue
                    elif issubclass(f.type, dict):  # standard dict
                        continue
                    else:
                        raise NotImplementedError(f.type)
                except Exception as e:
                    raise
                try:
                    setattr(self, f.name, dclass(**value))
                except Exception:
                    logger.error("Can't init %s (%s) with %s", f.name, dclass, value)
                    print(f.name, dclass, value)
                    raise

        # validate:
        for f in fields(self):
            value = getattr(self, f.name)
            try:
                if hasattr(f.type, "__origin__"):
                    if f.type.__origin__ is Union:
                        assert isinstance(value, f.type.__args__), (f.name, f.type.__args__)
                    else:
                        assert isinstance(value, f.type.__origin__), (f.name, f.type.__origin__)
                else:
                    assert isinstance(value, f.type), (f.name, f.type)
            except AssertionError:
                logger.error("Validation error for %s. expected %s", f.name, f.type)
                raise

class PeriodUnit(Enum):
    epoch = "epoch"
    iteration = "iteration"


@dataclass
class Period(ConfigBase):
    value: int
    unit: PeriodUnit

    def __post_init__(self):
        if isinstance(self.unit, str):
            self.unit = PeriodUnit(self.unit)

        super().__post_init__()


@dataclass
class ModelConfig(ConfigBase):
    name: str
    kwargs: Dict[str, Any]
    checkpoint: Optional[Path] = None

    def __post_init__(self):
        if isinstance(self.checkpoint, str):
            self.checkpoint = Path(self.checkpoint)

        assert self.checkpoint is None or self.checkpoint.exists()
        super().__post_init__()


@dataclass
class LogConfig(ConfigBase):
    log_scalars_period: Period
    log_images_period: Period
    log_bead_precision_recall: bool = False
    log_bead_precision_recall_threshold: float = 5.0
    save_n_checkpoints: int = 1


@dataclass
class TestConfig(ConfigBase):
    data: Dict[str, dict]
    batch_size: int
    transforms: List[dict]


@dataclass
class ValidateConfig(ConfigBase):
    data: Dict[str, dict]
    batch_size: int
    transforms: List[dict]
    period: Period


@dataclass
class LossConfig(ConfigBase):
    name: str
    kwargs: Dict[str, Any]


@dataclass
class OptimizerConfig(ConfigBase):
    name: str
    kwargs: Dict[str, Any]


@dataclass
class TrainConfig(ConfigBase):
    batch_size: int
    max_num_epochs: int
    log: LogConfig
    validate: ValidateConfig
    loss: LossConfig
    optimizer: OptimizerConfig
    score_metric: str
    patience: int
    batch_groups: Dict[str, dict]
    data: Dict[str, dict]


@dataclass
class Config(ConfigBase):
    precision: str
    nnum: int
    z_out: int

    model: ModelConfig
    test: TestConfig
    train: Optional[TrainConfig]
