import typer
from merge_args import merge_args

import hylfm  # noqa

import sys

from hylfm.get_model import get_model
from hylfm.hylfm_types import DatasetName

from hylfm.model import HyLFM_Net
from hylfm.run.train import TrainRun

app = typer.Typer()


@app.command()
@merge_args(get_model)
def train(dataset_name: DatasetName = typer.Option(..., "--dataset_name"),     batch_size: int = 1,
    interpolation_order: int = 2, **model_kwargs
):
    model = get_model(**model_kwargs)
    run = TrainRun()


if __name__ == "__main__":
    train()
