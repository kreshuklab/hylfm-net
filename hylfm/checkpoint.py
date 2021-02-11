import shutil
import sys
from dataclasses import asdict, dataclass, field, fields
from enum import Enum
from inspect import signature
from pathlib import Path
from typing import Collection, Dict, Optional, Union

import torch

from hylfm import __version__, settings
from hylfm.get_model import get_model
from hylfm.hylfm_types import (
    CriterionChoice,
    DatasetChoice,
    LRSchedThresMode,
    LRSchedulerChoice,
    MetricChoice,
    OptimizerChoice,
    PeriodUnit,
)


def conv_to_simple_dtypes(data: dict):
    return {
        k: v.value
        if isinstance(v, Enum)
        else str(v)
        if isinstance(v, Path)
        else conv_to_simple_dtypes(v)
        if isinstance(v, dict)
        else v
        for k, v in data.items()
    }


@dataclass
class RunConfig:
    batch_size: int
    data_range: float
    dataset: DatasetChoice
    interpolation_order: int
    win_sigma: float
    win_size: int
    save_output_to_disk: Optional[Collection[str]]

    def __post_init__(self):
        pass

    def as_dict(self, for_logging: bool = False) -> dict:
        dat = conv_to_simple_dtypes(asdict(self))
        return dat

    @classmethod
    def from_dict(cls, dat: dict):
        dat = dict(dat)
        return cls(dataset=DatasetChoice(dat.pop("dataset")), **dat)


@dataclass
class TrainRunConfig(RunConfig):
    batch_multiplier: int
    crit_apply_weight_above_threshold: bool
    crit_beta: float
    crit_decay_weight_by: Optional[float]
    crit_decay_weight_every_unit: PeriodUnit
    crit_decay_weight_every_value: int
    crit_decay_weight_limit: float
    crit_ms_ssim_weight: float
    crit_threshold: float
    crit_weight: float
    criterion: CriterionChoice
    eval_batch_size: int
    lr_sched_factor: float
    lr_sched_patience: int
    lr_sched_thres: float
    lr_sched_thres_mode: LRSchedThresMode
    lr_scheduler: Optional[LRSchedulerChoice]
    max_epochs: int
    model: Optional[Dict[str, Union[None, float, int, str]]]
    model_weights: Optional[Path]
    opt_lr: float
    opt_momentum: float
    opt_weight_decay: float
    optimizer: OptimizerChoice
    patience: int
    score_metric: MetricChoice
    seed: int
    validate_every_unit: PeriodUnit
    validate_every_value: int

    model_weights_name: Optional[str] = field(init=False)

    @classmethod
    def from_dict(cls, dat: dict):
        dat = dict(dat)

        # convert value types of super class
        super_key_words = signature(super().__init__).parameters
        super_dat = {k: v for k, v in dat.items() if k in super_key_words}
        super_dat = asdict(super().from_dict(super_dat))

        dat = {k: v for k, v in dat.items() if k not in super_key_words}
        mw = dat.pop("model_weights")
        lrs = dat.pop("lr_scheduler")
        return cls(
            crit_decay_weight_every_unit=PeriodUnit(dat.pop("crit_decay_weight_every_unit")),
            criterion=CriterionChoice(dat.pop("criterion")),
            lr_sched_thres_mode=LRSchedThresMode(dat.pop("lr_sched_thres_mode")),
            lr_scheduler=None if lrs is None else LRSchedulerChoice(lrs),
            model_weights=None if mw is None else Path(mw),
            optimizer=OptimizerChoice(dat.pop("optimizer")),
            score_metric=MetricChoice(dat.pop("score_metric")),
            validate_every_unit=PeriodUnit(dat.pop("validate_every_unit")),
            **dat,
            **super_dat,
        )

    def __post_init__(self):
        super().__post_init__()
        if self.model_weights is None:
            self.model_weights_name = None
        else:
            self.model_weights_name = self.model_weights.stem


@dataclass
class Checkpoint:
    config: TrainRunConfig
    training_run_id: str
    training_run_name: str

    best_validation_score: Optional[float] = None
    epoch: int = 0
    full_batch_len: int = None  # deprecated: todo: remove
    hylfm_version: str = __version__
    impatience: int = 0
    iteration: int = 0
    lr_scheduler_state_dict: Optional[dict] = None
    model_weights: Optional[dict] = None
    optimizer_state_dict: Optional[dict] = None
    validation_iteration: int = 0

    root: Path = field(init=False)

    def __post_init__(self):
        self.root = settings.log_dir / "checkpoints" / self.training_run_name
        self.root.mkdir(parents=True, exist_ok=True)

    @classmethod
    def load(cls, path: Path):
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        checkpoint_data = torch.load(str(path), map_location=device)
        if "model" in checkpoint_data:
            from hylfm.load_old_checkpoint import get_config_for_old_checkpoint

            # old checkpoint
            self = cls(
                config=get_config_for_old_checkpoint(path),
                hylfm_version="0.0.0",
                model_weights=checkpoint_data["model"],
                training_run_id="old_checkpoint",
                training_run_name=path.stem,
            )
        else:
            config = checkpoint_data.pop("config")
            if "nnum" in config:
                # someone was all about flattening that config dict, and somehow saved the flattened version in the checkpoint as well!?!?!!
                model_config = {}
                model_keys = signature(get_model).parameters
                for key in list(config.keys()):
                    if key in model_keys:
                        model_config[key] = config.pop(key)

                config["model"] = model_config

            config = TrainRunConfig.from_dict(config)
            self = cls(config=config, **checkpoint_data)

        return self

    @property
    def path(self):
        return self.root / f"val{self.validation_iteration:05}_ep{self.epoch}_it{self.iteration}.pth"

    def save(self, best: bool):
        assert self.training_run_name is not None
        assert self.training_run_id is not None
        path = self.path

        torch.save(self.as_dict(for_logging=False), path)

        if best:  # overwrite best
            best_path = self.root / "best.pth"
            try:
                best_path.unlink()
            except FileNotFoundError:
                pass
            # best_path.unlink(missing_ok=True)  # todo: python 3.8

            if sys.platform == "win32":
                shutil.copy(path, best_path)
            else:
                best_path.symlink_to(path)

        return path

    def as_dict(self, for_logging: bool) -> dict:
        dat = asdict(self)
        dat.pop("config")
        if for_logging:
            config_key = "cfg"
            for f in fields(self):
                if not f.init and not f.metadata.get("log", False):
                    dat.pop(f.name)
        else:
            config_key = "config"
            for f in fields(self):
                if not f.init:
                    dat.pop(f.name)

        dat = conv_to_simple_dtypes(dat)
        dat[config_key] = self.config.as_dict(for_logging=for_logging)
        return dat


@dataclass
class TestRunConfig(RunConfig):
    checkpoint: Optional[Checkpoint]

    def __post_init__(self):
        super().__post_init__()
        assert isinstance(self.checkpoint, Checkpoint), type(self.checkpoint)

    def as_dict(self, for_logging: bool = True):
        dat = super().as_dict(for_logging=for_logging)
        if dat.pop("checkpoint") is None:
            dat["cp"] = None
        else:
            dat["cp"] = self.checkpoint.path

        return dat
