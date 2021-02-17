import logging
from pathlib import Path
from pprint import pformat
from typing import List, Optional

import typer

from hylfm import settings
from hylfm.checkpoint import Checkpoint
from hylfm.hylfm_types import (
    CriterionChoice,
    DatasetChoice,
    LRSchedThresMode,
    LRSchedulerChoice,
    MetricChoice,
    OptimizerChoice,
    PeriodUnit,
)
from hylfm.train import train_from_checkpoint

app = typer.Typer()
logger = logging.getLogger(__name__)


@app.command()
def resume(
    checkpoint: Path,
    # optionally overwrite checkpoint
    impatience: Optional[int] = typer.Option(None, "--impatience"),
    best_validation_score: Optional[float] = typer.Option(None, "--best_validation_score"),
    reset_epoch: Optional[bool] = typer.Option(False, "--reset_epoch"),
    # optionally update config
    batch_multiplier: Optional[int] = typer.Option(None, "--batch_multiplier"),
    batch_size: Optional[int] = typer.Option(None, "--batch_size"),
    crit_decay: Optional[float] = typer.Option(None, "--crit_decay"),
    crit_decay_weight_every_unit: Optional[PeriodUnit] = typer.Option(None, "--crit_decay_weight_every_unit"),
    crit_decay_weight_every_value: Optional[int] = typer.Option(None, "--crit_decay_weight_every_value"),
    crit_threshold: Optional[float] = typer.Option(None, "--crit_threshold"),
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
    changes = {
        k: v
        for k, v in {
            "batch_multiplier": batch_multiplier,
            "batch_size": batch_size,
            "crit_decay": crit_decay,
            "crit_decay_weight_every_unit": crit_decay_weight_every_unit,
            "crit_decay_weight_every_value": crit_decay_weight_every_value,
            "crit_threshold": crit_threshold,
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

    checkpoint = checkpoint.resolve()
    checkpoint = Checkpoint.load(checkpoint)
    assert checkpoint.model_weights is not None, "what to resume from?"
    assert checkpoint.optimizer_state_dict is not None, "what to resume from?"
    assert checkpoint.lr_scheduler_state_dict is not None, "what to resume from?"

    if dataset is not None and checkpoint.config.dataset != dataset:
        checkpoint.config.dataset = dataset
        checkpoint.iteration = 0
        if impatience is None:
            impatience = 0

    if reset_epoch:
        changes["epoch"] = 0
        checkpoint.epoch = 0

    if impatience is not None:
        changes["impatience"] = impatience
        checkpoint.impatience = impatience

    if best_validation_score is not None:
        changes["best_validation_score"] = best_validation_score

    note = f"resume to train {checkpoint} {'with changes: ' + pformat(changes) if changes else ''}"
    logger.info(note)

    config = checkpoint.config.as_dict(for_logging=False)
    config["resumed_from"] = checkpoint.training_run_name

    import wandb

    wandb_run = wandb.init(
        project="HyLFM-train", dir=str(settings.cache_dir), config=config, resume="allow", notes=note
    )
    checkpoint.training_run_name = wandb_run.name
    checkpoint.training_run_id = wandb_run.id

    train_from_checkpoint(wandb_run, checkpoint)


if __name__ == "__main__":
    app()
