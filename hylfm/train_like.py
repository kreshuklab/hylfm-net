import logging
from pathlib import Path
from typing import List, Optional

import typer

from hylfm import __version__
from hylfm.checkpoint import Checkpoint
from hylfm.hylfm_types import (
    CriterionChoice,
    DatasetChoice,
    LRSchedThresMode,
    LRSchedulerChoice,
    MetricChoice,
    OptimizerChoice,
)
from hylfm.train import train
from hylfm.utils.general import PeriodUnit

app = typer.Typer()


logger = logging.getLogger(__name__)


@app.command()
def train_model_like(
    model_kwargs_from_checkpoint: Path,
    batch_multiplier: Optional[int] = typer.Option(None, "--batch_multiplier"),
    batch_size: Optional[int] = typer.Option(None, "--batch_size"),
    crit_weight: Optional[float] = typer.Option(None, "--crit_weight"),
    criterion: Optional[CriterionChoice] = typer.Option(None, "--criterion"),
    dataset: Optional[DatasetChoice] = typer.Option(None, "--dataset"),
    eval_batch_size: Optional[int] = typer.Option(None, "--eval_batch_size"),
    interpolation_order: Optional[int] = typer.Option(None, "--interpolation_order"),
    lr_sched_factor: Optional[float] = typer.Option(None, "--lr_sched_factor"),
    lr_sched_patience: Optional[int] = typer.Option(None, "--lr_sched_patience"),
    lr_sched_thres: Optional[float] = typer.Option(None, "--lr_sched_thres"),
    lr_sched_thres_mode: Optional[LRSchedThresMode] = typer.Option(None, "--lr_sched_thres_mode"),
    lr_scheduler: Optional[LRSchedulerChoice] = typer.Option(None, "--lr_scheduler"),
    max_epochs: Optional[int] = typer.Option(None, "--max_epochs"),
    model_weights: Optional[Path] = typer.Option(None, "--model_weights"),
    opt_lr: Optional[float] = typer.Option(None, "--opt_lr"),
    opt_momentum: Optional[float] = typer.Option(None, "--opt_momentum"),
    opt_weight_decay: Optional[float] = typer.Option(None, "--opt_weight_decay"),
    optimizer: Optional[OptimizerChoice] = typer.Option(None, "--optimizer"),
    patience: Optional[int] = typer.Option(None, "--patience"),
    save_after_validation_iterations: Optional[List[int]] = typer.Option(None, "--save_after_validation_iterations"),
    score_metric: MetricChoice = typer.Option(MetricChoice.MS_SSIM, "--score_metric"),
    seed: Optional[int] = typer.Option(None, "--seed"),
    validate_every_unit: Optional[PeriodUnit] = typer.Option(None, "--validate_every_unit"),
    validate_every_value: Optional[int] = typer.Option(None, "--validate_every_value"),
    win_sigma: Optional[float] = typer.Option(None, "--win_sigma"),
    win_size: Optional[int] = typer.Option(None, "--win_size"),
    z_out: Optional[int] = typer.Option(None, "--z_out"),
    zero_max_patience: Optional[int] = typer.Option(None, "--zero_max_patience"),
):
    reference_checkpoint = Checkpoint.load(model_kwargs_from_checkpoint)
    reference_config = reference_checkpoint.config
    config = reference_config.as_dict(for_logging=False)
    config.pop("hylfm_version")
    config.update(config.pop("model"))  # flatten model kwargs into config

    changes = {
        k: v
        for k, v in {
            "batch_multiplier": batch_multiplier,
            "batch_size": batch_size,
            "crit_weight": crit_weight,
            "criterion": criterion,
            "dataset": dataset,
            "eval_batch_size": eval_batch_size,
            "interpolation_order": interpolation_order,
            "lr_sched_factor": lr_sched_factor,
            "lr_sched_patience": lr_sched_patience,
            "lr_sched_thres": lr_sched_thres,
            "lr_sched_thres_mode": lr_sched_thres_mode,
            "lr_scheduler": lr_scheduler,
            "max_epochs": max_epochs,
            "model_weights": model_weights,
            "opt_lr": opt_lr,
            "opt_momentum": opt_momentum,
            "opt_weight_decay": opt_weight_decay,
            "optimizer": optimizer,
            "patience": patience,
            "save_after_validation_iterations": save_after_validation_iterations,
            "score_metric": score_metric,
            "seed": seed,
            "validate_every_unit": validate_every_unit,
            "validate_every_value": validate_every_value,
            "win_sigma": win_sigma,
            "win_size": win_size,
            "z_out": z_out,
            "zero_max_patience": zero_max_patience,
        }.items()
        if v is not None
    }

    note = f"train like {model_kwargs_from_checkpoint.resolve()} {'with changes:' if changes else ''} " + " ".join(
        [f"{k}: {v}" for k, v in changes.items()]
    )

    config.update(changes)
    train(**config, note=note)


if __name__ == "__main__":
    app()
