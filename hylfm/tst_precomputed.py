import logging
from pathlib import Path
from typing import Optional

from hylfm import __version__, settings  # import hylfm before numpy
from hylfm.checkpoint import RunConfig, TestPrecomputedRunConfig
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


def get_save_output_to_disk_from_path(log_level_disk: int, source_path: Path, ui_name: str):
    tensors_to_log = ["metrics", "pred", "spim", "lf"]

    save_output_to_disk = {}
    for lvl, key in enumerate(tensors_to_log):
        if lvl >= log_level_disk:
            break

        on_disk_name = key + ".h5" if key == "metrics" else key
        save_to = source_path / ui_name / on_disk_name
        if save_to.exists():
            raise FileExistsError(save_to)

        save_output_to_disk[key] = save_to

    return save_output_to_disk


@app.command(name="test_precomputed")
def tst_precomputed(
    pred: str,
    batch_size: int = typer.Option(1, "--batch_size"),
    data_range: float = typer.Option(1, "--data_range"),
    dataset: Optional[DatasetChoice] = DatasetChoice.from_path,
    from_path: Optional[Path] = typer.Option(None, "--from_path"),
    pred_glob: Optional[str] = typer.Option(None, "--pred_glob"),
    trgt_glob: Optional[str] = typer.Option(None, "--trgt_glob"),
    interpolation_order: int = typer.Option(2, "--interpolation_order"),
    log_level_disk: Optional[int] = typer.Option(None, "--log_level_disk"),
    log_level_wandb: int = typer.Option(1, "--log_level_wandb"),
    point_cloud_threshold: float = typer.Option(1.0, "--point_cloud_threshold"),
    scale: int = 4,
    shrink: int = 8,
    trgt: Optional[str] = None,
    ui_name: Optional[str] = typer.Option(None, "--ui_name"),
    win_sigma: float = typer.Option(1.5, "--win_sigma"),
    win_size: int = typer.Option(11, "--win_size"),
):

    # todo: split if into different functions / run configs / runs??
    if dataset == DatasetChoice.from_path:
        if log_level_disk is not None:
            logger.warning("ignoring log_level_disk %s for data from disk at %s", log_level_disk, from_path)

        log_level_disk = 1

        logger.warning("ignoring scale %s and shrink %s for testing precomputed from path %s", scale, shrink, from_path)
        scale = 1
        shrink = 0
        if from_path is None:
            raise ValueError("missing input 'from_path'")

        if trgt is None:
            raise ValueError("missing input 'trgt'")

        if pred_glob is None:
            raise ValueError("missing input 'pred_glob'")

        if trgt_glob is None:
            raise ValueError("missing input 'trgt_glob'")

        if ui_name is None:
            ui_name = f"{pred}_vs_{trgt}"

        save_output_to_disk = get_save_output_to_disk_from_path(log_level_disk, from_path, ui_name)
    else:
        if log_level_disk is None:
            log_level_disk = 2

        assert shrink
        if from_path is not None:
            raise ValueError(f"don't specify 'from_path' when testing on defined dataset {dataset}")

        if trgt is not None:
            raise ValueError(f"don't specify 'trgt' when testing on defined dataset {dataset}")

        if ui_name is None:
            ui_name = pred

        save_output_to_disk = get_save_output_to_disk(log_level_disk, dataset, ui_name)

    config = TestPrecomputedRunConfig(
        path=from_path,
        pred_name=pred,
        pred_glob=pred_glob,
        trgt_name=trgt,
        trgt_glob=trgt_glob,
        batch_size=batch_size,
        data_range=data_range,
        dataset=dataset,
        interpolation_order=interpolation_order,
        win_sigma=win_sigma,
        win_size=win_size,
        save_output_to_disk=save_output_to_disk,
        hylfm_version=__version__,
        point_cloud_threshold=point_cloud_threshold,
    )

    import wandb

    wandb_run = wandb.init(
        project=f"HyLFM-test", dir=str(settings.cache_dir), config=config.as_dict(for_logging=True), name=ui_name
    )

    test_run = TestPrecomputedRun(
        config=config, wandb_run=wandb_run, log_level_wandb=log_level_wandb, scale=scale, shrink=shrink
    )

    test_run.run()


if __name__ == "__main__":
    app()
