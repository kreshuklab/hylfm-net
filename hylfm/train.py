from hylfm import __version__, metrics, settings  # noqa: first line to set numpy env vars

import logging
import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy
import torch.optim
import typer
import wandb
from merge_args import merge_args

from hylfm.checkpoint import Checkpoint, TrainRunConfig
from hylfm.get_model import get_model
from hylfm.hylfm_types import (
    CriterionChoice,
    DatasetChoice,
    LRSchedThresMode,
    LRSchedulerChoice,
    MetricChoice,
    OptimizerChoice,
)
from hylfm.run.train_run import TrainRun
from hylfm.utils.general import PeriodUnit

app = typer.Typer()


logger = logging.getLogger(__name__)


@app.command()
@merge_args(get_model)
def train(
    dataset: DatasetChoice,
    batch_multiplier: int = typer.Option(1, "--batch_multiplier"),
    batch_size: int = typer.Option(1, "--batch_size"),
    crit_apply_weight_above_threshold: bool = typer.Option(False, "--crit_apply_weight_above_threshold"),
    crit_beta: float = typer.Option(1.0, "--crit_beta"),
    crit_decay_weight_by: Optional[float] = typer.Option(None, "--crit_decay_weight_by"),
    crit_decay_weight_every_unit: PeriodUnit = typer.Option(PeriodUnit.epoch, "--crit_decay_weight_every_unit"),
    crit_decay_weight_every_value: int = typer.Option(1, "--crit_decay_weight_every_value"),
    crit_decay_weight_limit: float = typer.Option(1.0, "--crit_decay_weight_limit"),
    crit_ms_ssim_weight: float = typer.Option(0.001, "--crit_ms_ssim_weight"),
    crit_threshold: float = typer.Option(1.0, "--crit_threshold"),
    crit_weight: float = typer.Option(0.05, "--crit_weight"),
    criterion: CriterionChoice = typer.Option(CriterionChoice.L1, "--criterion"),
    data_range: float = typer.Option(1.0, "--data_range"),
    eval_batch_size: int = typer.Option(1, "--eval_batch_size"),
    interpolation_order: int = typer.Option(2, "--interpolation_order"),
    lr_sched_factor: float = typer.Option(0.1, "--lr_sched_factor"),
    lr_sched_patience: int = typer.Option(10, "--lr_sched_patience"),
    lr_sched_thres: float = typer.Option(0.0001, "--lr_sched_thres"),
    lr_sched_thres_mode: LRSchedThresMode = typer.Option(LRSchedThresMode.abs, "--lr_sched_thres_mode"),
    lr_scheduler: Optional[LRSchedulerChoice] = typer.Option(None, "--lr_scheduler"),
    max_epochs: int = typer.Option(10, "--max_epochs"),
    model_weights: Optional[Path] = typer.Option(None, "--model_weights"),
    opt_lr: float = typer.Option(1e-4, "--opt_lr"),
    opt_momentum: float = typer.Option(0.0, "--opt_momentum"),
    opt_weight_decay: float = typer.Option(0.0, "--opt_weight_decay"),
    optimizer: OptimizerChoice = typer.Option(OptimizerChoice.Adam, "--optimizer"),
    patience: int = typer.Option(5, "--patience"),
    score_metric: MetricChoice = typer.Option(MetricChoice.MS_SSIM, "--score_metric"),
    seed: Optional[int] = typer.Option(None, "--seed"),
    validate_every_unit: PeriodUnit = typer.Option(PeriodUnit.epoch, "--validate_every_unit"),
    validate_every_value: int = typer.Option(1, "--validate_every_value"),
    win_sigma: float = typer.Option(1.5, "--win_sigma"),
    win_size: int = typer.Option(11, "--win_size"),
    **model_kwargs,
):
    config = TrainRunConfig(
        batch_multiplier=batch_multiplier,
        batch_size=batch_size,
        crit_apply_weight_above_threshold=crit_apply_weight_above_threshold,
        crit_beta=crit_beta,
        crit_decay_weight_by=crit_decay_weight_by,
        crit_decay_weight_every_unit=crit_decay_weight_every_unit,
        crit_decay_weight_every_value=crit_decay_weight_every_value,
        crit_decay_weight_limit=crit_decay_weight_limit,
        crit_ms_ssim_weight=crit_ms_ssim_weight,
        crit_threshold=crit_threshold,
        crit_weight=crit_weight,
        criterion=criterion,
        data_range=data_range,
        dataset=dataset,
        eval_batch_size=eval_batch_size,
        interpolation_order=interpolation_order,
        lr_sched_factor=lr_sched_factor,
        lr_sched_patience=lr_sched_patience,
        lr_sched_thres=lr_sched_thres,
        lr_sched_thres_mode=lr_sched_thres_mode,
        lr_scheduler=lr_scheduler,
        max_epochs=max_epochs,
        model=model_kwargs,
        model_weights=model_weights,
        opt_lr=opt_lr,
        opt_momentum=opt_momentum,
        opt_weight_decay=opt_weight_decay,
        optimizer=optimizer,
        patience=patience,
        save_output_to_disk={},
        score_metric=score_metric,
        seed=seed,
        validate_every_unit=validate_every_unit,
        validate_every_value=validate_every_value,
        win_sigma=win_sigma,
        win_size=win_size,
        hylfm_version=__version__,
        point_cloud_threshold=1.0,
    )

    wandb_run = wandb.init(project="HyLFM-train", dir=str(settings.cache_dir), config=config.as_dict(for_logging=True))

    if model_weights is not None:
        model_weights = Checkpoint.load(model_weights).model_weights

    checkpoint = Checkpoint(
        model_weights=model_weights, config=config, training_run_name=wandb_run.name, training_run_id=wandb_run.id
    )

    train_from_checkpoint(wandb_run, checkpoint=checkpoint)


def train_from_checkpoint(wandb_run, checkpoint: Checkpoint):
    cfg = checkpoint.config
    if cfg.seed is not None:
        numpy.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

    train_run = TrainRun(wandb_run=wandb_run, checkpoint=checkpoint)
    train_run.fit()

    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).parent / "tst.py"),
            "--wandb_logging",
            1,
            str(checkpoint.path.with_name("best.pth")),
            # str(checkpoint.path.with_stem("best")),  # todo: python 3.9
        ]
    )


if __name__ == "__main__":
    app()
