import collections
from pathlib import Path
from typing import Optional

import typer

from hylfm import settings
from hylfm.checkpoint import Checkpoint
from hylfm.hylfm_types import DatasetChoice, PeriodUnit
from hylfm.train import train_from_checkpoint

app = typer.Typer()


@app.command()
def resume(
    checkpoint: Path,
    best_validation_score: Optional[float] = typer.Option(None, "--best_validation_score"),
    dataset: Optional[DatasetChoice] = None,
    impatience: Optional[int] = typer.Option(None, "--impatience"),
    max_epochs: Optional[int] = typer.Option(None, "--max_epochs"),
    patience: Optional[int] = typer.Option(None, "--patience"),
    reset_epoch: Optional[bool] = typer.Option(False, "--reset_epoch"),
    validate_every_unit: Optional[PeriodUnit] = typer.Option(None, "--validate_every_unit"),
    validate_every_value: Optional[int] = typer.Option(None, "--validate_every_value"),
    eval_batch_size: Optional[int] = typer.Option(None, "--eval_batch_size"),
):
    checkpoint = Checkpoint.load(checkpoint)

    changes = collections.OrderedDict()
    if dataset is not None and checkpoint.config.dataset != dataset:
        checkpoint.config.dataset = dataset
        checkpoint.iteration = 0
        changes["dataset"] = dataset.value
        if impatience is None:
            impatience = 0

    if eval_batch_size is not None:
        changes["eval_batch_size"] = f"{checkpoint.config.eval_batch_size}->{eval_batch_size}"
        checkpoint.config.eval_batch_size = eval_batch_size

    if best_validation_score is not None:
        changes["best_validation_score"] = f"{checkpoint.best_validation_score}->{best_validation_score}"
        checkpoint.best_validation_score = best_validation_score

    if max_epochs is not None:
        changes["max_epochs"] = f"{checkpoint.config.max_epochs}->{max_epochs}"
        checkpoint.config.max_epochs = max_epochs

    if reset_epoch:
        changes["epoch"] = f"{checkpoint.epoch}->0"
        checkpoint.epoch = 0

    if impatience is not None:
        changes["impatience"] = f"{checkpoint.impatience}->{impatience}"
        checkpoint.impatience = impatience

    if patience is not None:
        changes["patience"] = f"{checkpoint.config.patience}->{patience}"
        checkpoint.config.patience = patience

    if validate_every_unit is not None:
        changes["validate_every_unit"] = f"{checkpoint.config.validate_every_unit}->{validate_every_unit}"
        checkpoint.config.validate_every_unit = validate_every_unit

    if validate_every_value is not None:
        changes["validate_every_value"] = f"{checkpoint.config.validate_every_value}->{validate_every_value}"
        checkpoint.config.validate_every_value = validate_every_value

    if changes:
        notes = "resumed with changes: " + " ".join([f"{k}: {v}" for k, v in changes.items()])
    else:
        notes = "resumed without changes"

    import wandb

    config = checkpoint.config.as_dict(for_logging=False)
    config["resumed_from"] = checkpoint.training_run_name

    wandb_run = wandb.init(
        project="HyLFM-train", dir=str(settings.cache_dir), config=config, resume="allow", notes=notes
    )
    checkpoint.training_run_name = wandb_run.name
    checkpoint.training_run_id = wandb_run.id

    train_from_checkpoint(wandb_run, checkpoint)


if __name__ == "__main__":
    app()
