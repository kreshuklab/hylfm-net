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
):
    checkpoint = Checkpoint.load(checkpoint)

    changes = collections.OrderedDict()
    if dataset is not None and checkpoint.config.dataset != dataset:
        checkpoint.config.dataset = dataset
        checkpoint.iteration = 0
        changes["dataset"] = dataset.value
        if impatience is None:
            impatience = 0

    if best_validation_score is not None:
        checkpoint.best_validation_score = best_validation_score
        changes["best_validation_score"] = best_validation_score

    if max_epochs is not None:
        checkpoint.config.max_epochs = max_epochs
        changes["max_epochs"] = max_epochs

    if reset_epoch:
        checkpoint.epoch = 0
        changes["epoch"] = 0

    if impatience is not None:
        checkpoint.impatience = impatience
        changes["impatience"] = impatience

    if patience is not None:
        checkpoint.patience = patience
        changes["patience"] = patience

    if validate_every_unit is not None:
        checkpoint.config.validate_every_unit = validate_every_unit
        changes["validate_every_unit"] = validate_every_unit

    if validate_every_value is not None:
        checkpoint.config.validate_every_value = validate_every_value
        changes["validate_every_value"] = validate_every_value

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
