import shutil
import sys
from dataclasses import asdict, dataclass, field, fields
from enum import Enum
from inspect import signature
from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import packaging.version
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


def _conv_to_simple_dtypes(data: dict):
    return {
        k: v.value
        if isinstance(v, Enum)
        else str(v)
        if isinstance(v, Path)
        else _conv_to_simple_dtypes(v)
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
    save_output_to_disk: Optional[Dict[str, Path]]
    point_cloud_threshold: float
    hylfm_version: str

    def __post_init__(self):
        pass

    def as_dict(self, for_logging: bool = False) -> dict:
        dat = _conv_to_simple_dtypes(asdict(self))
        return dat

    @classmethod
    def add_new_keys_for_0_1_2(cls, dat: dict) -> dict:
        if "hylfm_version" not in dat:
            dat["hylfm_version"] = "0.1.2.elevated"

        if "save_output_to_disk" not in dat:
            dat["save_output_to_disk"] = None

        if "point_cloud_threshold" not in dat:
            dat["point_cloud_threshold"] = 1.0

        return dat

    @classmethod
    def convert_dict(cls, dat: dict) -> dict:
        dat = dict(dat)
        dat["dataset"] = DatasetChoice(dat.pop("dataset"))
        return dat

    @classmethod
    def from_dict(cls, dat: dict):
        dat = dict(dat)
        if packaging.version.parse(dat.get("hylfm_version", "0.0.0")) < packaging.version.parse("0.1.2"):
            dat = cls.add_new_keys_for_0_1_2(dat)

        dat = cls.convert_dict(dat)

        return cls(**dat)


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
    save_after_validation_iterations: Sequence[int] = tuple()

    model_weights_name: Optional[str] = None
    zero_max_patience: int = 10

    def as_dict(self, for_logging: bool = False) -> dict:
        dat = super().as_dict(for_logging=for_logging)
        if for_logging:
            dat.update(dat.pop("model"))
        else:
            dat.pop("model_weights_name")

        return dat

    @classmethod
    def convert_dict(cls, dat: dict):
        dat = super().convert_dict(dat)
        dat["crit_decay_weight_every_unit"] = PeriodUnit(dat.pop("crit_decay_weight_every_unit", "epoch"))
        dat["criterion"] = CriterionChoice(dat.pop("criterion"))
        dat["lr_sched_thres_mode"] = LRSchedThresMode(dat.pop("lr_sched_thres_mode"))
        lrs = dat.pop("lr_scheduler")
        dat["lr_scheduler"] = None if lrs is None else LRSchedulerChoice(lrs)
        mw = dat.pop("model_weights")
        dat["model_weights"] = None if mw is None else Path(mw)
        dat["optimizer"] = OptimizerChoice(dat.pop("optimizer"))
        dat["score_metric"] = MetricChoice(dat.pop("score_metric"))
        dat["validate_every_unit"] = PeriodUnit(dat.pop("validate_every_unit"))

        return dat

    @classmethod
    def add_new_keys_for_0_1_2(cls, dat: dict):
        dat = super().add_new_keys_for_0_1_2(dat)
        for key in ["lr_sched_factor", "lr_sched_patience", "lr_sched_thres", "lr_scheduler"]:
            if key not in dat:
                dat[key] = None

        if "lr_sched_thres_mode" not in dat:
            dat["lr_sched_thres_mode"] = LRSchedThresMode.abs

        if "score_metric" not in dat:
            dat["score_metric"] = MetricChoice.MS_SSIM.value

        return dat

    def __post_init__(self):
        super().__post_init__()
        if self.model_weights is not None and self.model_weights_name is None:
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

            # todo: remove this monkey path
            savi = config.pop("save_after_validation_iterations", [])
            if isinstance(savi, list):
                config["save_after_validation_iterations"] = savi
            else:
                # somehow a 'typer.models.OptionInfo' found its way here...
                config["save_after_validation_iterations"] = []

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
            # best_path = self.path.with_stem("best")  # todo: python 3.9
            best_path = self.path.with_name("best.pth")
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
            for key in ["lr_scheduler_state_dict", "model_weights", "optimizer_state_dict"]:
                dat.pop(key)
        else:
            config_key = "config"
            for f in fields(self):
                if not f.init:
                    dat.pop(f.name)

        dat = _conv_to_simple_dtypes(dat)
        dat[config_key] = self.config.as_dict(for_logging=for_logging)
        return dat


@dataclass
class ValidationRunConfig(RunConfig):
    """validate a model on a defined dataset"""

    pass


@dataclass
class TestCheckpointRunConfig(RunConfig):
    checkpoint: Checkpoint

    def __post_init__(self):
        super().__post_init__()
        assert isinstance(self.checkpoint, Checkpoint), type(self.checkpoint)

    def as_dict(self, for_logging: bool = True):
        dat = super().as_dict(for_logging=for_logging)
        checkpoint_key = "cp" if for_logging else "checkpoint"
        if dat.pop("checkpoint") is None:
            dat[checkpoint_key] = None
        else:
            dat[checkpoint_key] = self.checkpoint.path

        return dat

    @classmethod
    def convert_dict(cls, dat: dict):
        dat = super().convert_dict(dat)
        dat["checkpoint"] = Checkpoint.load(Path(dat["checkpoint"]))
        return dat


@dataclass
class PredictPathRunConfig(TestCheckpointRunConfig):
    path: Path
    glob_lf: str


@dataclass
class TestPrecomputedRunConfig(RunConfig):
    pred_name: str
    path: Optional[Path] = None
    pred_glob: Optional[str] = None
    trgt_name: Optional[str] = None
    trgt_glob: Optional[str] = None

    def __post_init__(self):
        inputs_only_for_path = {ipt: getattr(self, ipt) for ipt in ["path", "pred_glob", "trgt_glob"]}
        if self.dataset == DatasetChoice.from_path:
            invalid_inputs = {ipt: value for ipt, value in inputs_only_for_path.items() if value is None}
            if self.trgt_name is None:
                invalid_inputs["target_name"] = None
        else:
            invalid_inputs = {ipt for ipt, value in inputs_only_for_path.items() if value is not None}

        if invalid_inputs:
            raise ValueError(f"invalid inputs for {self.dataset}: {inputs_only_for_path}")
