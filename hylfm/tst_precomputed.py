import logging
from typing import Optional

from hylfm import __version__, settings  # import hylfm before numpy
from hylfm.checkpoint import RunConfig
from hylfm.datasets.named import DatasetChoice
from hylfm.run.eval_run import TestPrecomputedRun
from hylfm.tst import get_save_output_to_disk

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import typer

logger = logging.getLogger(__name__)


app = typer.Typer()


@app.command(name="test_precomputed")
def tst_precomputed(
    dataset: DatasetChoice,
    pred: str,
    batch_size: int = typer.Option(1, "--batch_size"),
    data_range: float = typer.Option(1, "--data_range"),
    interpolation_order: int = typer.Option(2, "--interpolation_order"),
    scale: int = 4,
    shrink: int = 0,
    ui_name: Optional[str] = typer.Option(None, "--ui_name"),
    win_sigma: float = typer.Option(1.5, "--win_sigma"),
    win_size: int = typer.Option(11, "--win_size"),
    point_cloud_threshold: float = typer.Option(1.0, "--point_cloud_threshold"),
    log_level_disk: int = typer.Option(2, "--log_level_disk"),
):

    if ui_name is None:
        ui_name = pred

    config = RunConfig(
        batch_size=batch_size,
        data_range=data_range,
        dataset=dataset,
        interpolation_order=interpolation_order,
        win_sigma=win_sigma,
        win_size=win_size,
        save_output_to_disk=get_save_output_to_disk(log_level_disk, dataset, ui_name),
        hylfm_version=__version__,
        point_cloud_threshold=point_cloud_threshold,
    )

    import wandb

    wandb_run = wandb.init(
        project=f"HyLFM-test", dir=str(settings.cache_dir), config=config.as_dict(for_logging=True), name=ui_name
    )

    test_run = TestPrecomputedRun(
        config=config, wandb_run=wandb_run, pred_name=pred, log_level_wandb=1, scale=scale, shrink=shrink
    )

    test_run.run()


if __name__ == "__main__":
    app()
