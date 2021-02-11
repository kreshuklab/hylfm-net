from pathlib import Path
from typing import Optional

import typer

from hylfm import settings
from hylfm.checkpoint import Checkpoint
from hylfm.train import train_from_checkpoint

app = typer.Typer()


@app.command()
def resume(checkpoint: Path, impatience: Optional[int] = typer.Option(None, "--impatience")):

    checkpoint = Checkpoint.load(checkpoint)
    if impatience is not None:
        checkpoint.impatience = impatience

    import wandb

    wandb_run = wandb.init(
        project="HyLFM-train",
        dir=str(settings.cache_dir),
        config=checkpoint.config.as_dict(checkpoint.model),
        resume="must",
        name=checkpoint.training_run_name,
        id=checkpoint.training_run_id,
    )
    train_from_checkpoint(wandb_run, checkpoint)


if __name__ == "__main__":
    app()
