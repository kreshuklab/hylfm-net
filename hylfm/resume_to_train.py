import collections
from pathlib import Path
from typing import Optional

import typer

from hylfm import settings
from hylfm.checkpoint import Checkpoint
from hylfm.hylfm_types import DatasetChoice
from hylfm.train import train_from_checkpoint

app = typer.Typer()


@app.command()
def resume(
    checkpoint: Path,
    dataset: Optional[DatasetChoice] = None,
    impatience: Optional[int] = typer.Option(None, "--impatience"),
):

    checkpoint = Checkpoint.load(checkpoint)

    changes = collections.OrderedDict()
    if dataset is not None:
        checkpoint.config.dataset = dataset
        changes["dataset"] = dataset.value

    if impatience is not None:
        checkpoint.impatience = impatience
        changes["impatience"] = impatience

    if changes:
        notes = "resumed with changes: " + " ".join([f"{k}: {v}" for k, v in changes.items()])
    else:
        notes = "resumed without changes"

    import wandb

    config = checkpoint.as_dict(for_logging=True)
    config["resumed_from"] = checkpoint.training_run_name

    checkpoint.training_run_name = None
    checkpoint.training_run_id = None

    wandb_run = wandb.init(
        project="HyLFM-train",
        dir=str(settings.cache_dir),
        config=config,
        resume="must",
        name=checkpoint.training_run_name,
        notes=notes,
        id=checkpoint.training_run_id,
    )
    train_from_checkpoint(wandb_run, checkpoint)


if __name__ == "__main__":
    app()
