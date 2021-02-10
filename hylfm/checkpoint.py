import shutil
import sys
from dataclasses import InitVar, asdict, dataclass, field, fields
from enum import Enum
from inspect import signature
from pathlib import Path
from typing import Dict, Optional, Sequence, Type, Union

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

import hylfm.metrics
from hylfm import __version__, metrics, settings
from hylfm.datasets import ConcatDataset, get_collate
from hylfm.datasets.named import get_dataset
from hylfm.get_criterion import get_criterion
from hylfm.get_model import get_model
from hylfm.hylfm_types import (
    CriterionChoice,
    DatasetChoice,
    DatasetPart,
    LRSchedThresMode,
    LRScheduler,
    LRSchedulerChoice,
    MetricChoice,
    Optimizer,
    OptimizerChoice,
    PeriodUnit,
    TransformsPipeline,
)
from hylfm.metrics import MetricGroup
from hylfm.model import HyLFM_Net
from hylfm.sampler import NoCrossBatchSampler
from hylfm.transform_pipelines import get_transforms_pipeline


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
class Config:
    batch_multiplier: int
    batch_size: int
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
    data_range: float
    dataset: DatasetChoice
    eval_batch_size: int
    interpolation_order: int
    lr_sched_factor: float
    lr_sched_patience: int
    lr_sched_thres: float
    lr_sched_thres_mode: LRSchedThresMode
    lr_scheduler: LRSchedulerChoice
    lr_scheduler: Optional[LRSchedulerChoice]
    max_epochs: int
    model: Dict[str, Union[None, float, int, str]]
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
    win_sigma: float
    win_size: int

    model_weights_name: Optional[str] = field(init=False)

    def as_dict(self, for_logging: bool) -> dict:
        dat = conv_to_simple_dtypes(asdict(self))
        if not for_logging:
            dat.pop("model_weights_name")
            dat.update(dat.pop("model"))  # flatten nested model dict

        return dat

    @classmethod
    def from_dict(cls, dat: dict):
        dat = dict(dat)
        mw = dat.pop("model_weights")
        return cls(
            criterion=CriterionChoice(dat.pop("criterion")),
            dataset=DatasetChoice(dat.pop("dataset")),
            model_weights=None if mw is None else Path(mw),
            optimizer=OptimizerChoice(dat.pop("optimizer")),
            validate_every_unit=PeriodUnit(dat.pop("validate_every_unit")),
            **dat,
        )

    def __post_init__(self):
        if self.model_weights is None:
            self.model_weights_name = None
        else:
            self.model_weights_name = self.model_weights.stem

@dataclass
class IndependentConfig:
    light_logging: bool

@dataclass
class Checkpoint:
    config: Config
    training_run_name: str
    training_run_id: str
    full_batch_len: int = None  # deprecated: todo: remove
    best_validation_score: Optional[float] = None
    epoch: int = 0
    impatience: int = 0
    iteration: int = 0
    validation_iteration: int = 0
    hylfm_version: str = __version__

    model_weights: Optional[dict] = None
    opt_state_dict: Optional[dict] = None
    lr_scheduler_state_dict: Optional[dict] = None

    root: Path = field(init=False)


    @classmethod
    def load(cls, path: Path):
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        checkpoint_data = torch.load(str(path), map_location=device)
        if "model" in checkpoint_data:
            from hylfm.load_old_checkpoint import get_config_for_old_checkpoint

            # old checkpoint
            return cls(
                config=get_config_for_old_checkpoint(path),
                hylfm_version="0.0.0",
                model_weights=checkpoint_data["model"],
                training_run_id=None,
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

            config = Config.from_dict(config)
            return cls(config=config, **checkpoint_data)


    def save(self, best: bool, keep_anyway: bool):
        assert self.training_run_name is not None
        assert self.training_run_id is not None
        if best or keep_anyway:
            path = self.root / f"val{self.validation_iteration:05}_ep{self.epoch}_it{self.iteration}.pth"
            best_path = self.root / "best.pth"
            if best:
                try:
                    best_path.unlink()
                except FileNotFoundError:
                    pass
                # best_path.unlink(missing_ok=True)  # todo: python 3.8

                # remove old best
                if self.current_best_on_disk is not None:
                    self.current_best_on_disk.unlink()
                    self.current_best_on_disk = None

            if not keep_anyway:
                # remember current best to delete on finding new best
                self.current_best_on_disk = path

            torch.save(self.as_dict(for_logging=False), path)

            if best:
                if sys.platform == "win32":
                    shutil.copy(path, best_path)
                    # (best_path).link_to(path)  # todo: python 3.8
                else:
                    (best_path).symlink_to(path)

    def as_dict(self, for_logging: bool) -> dict:
        dat = asdict(self)
        dat.pop("config")
        if for_logging:
            config_key = "cfg"
            for f in fields(self):
                if not f.init and not f.metadata.get("log", False):
                    dat.pop(f.name)
        else:
            dat["model_weights"] = self.model.state_dict()
            config_key = "config"
            for f in fields(self):
                if not f.init:
                    dat.pop(f.name)

        dat[config_key] = self.config.as_dict(for_logging=for_logging)
        return conv_to_simple_dtypes(dat)


@dataclass
class TestConfig:
    batch_size: int
    checkpoint: Optional[Checkpoint]
    data_range: float
    dataset: DatasetChoice
    dataset_part: DatasetPart
    interpolation_order: int
    win_sigma: float
    win_size: int

    def as_dict(self):
        dat = asdict(self)
        if dat.pop("checkpoint") is not None:
            dat["cp"] = self.checkpoint.as_dict(for_logging=True)

        return conv_to_simple_dtypes(dat)
