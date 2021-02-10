from hylfm import metrics, settings  # noqa: first line to set numpy env vars

import logging
from pathlib import Path
from typing import Optional

from torch.utils.data import DataLoader, SequentialSampler

from hylfm.checkpoint import Checkpoint, TestConfig
from hylfm.datasets import get_collate
from hylfm.datasets.named import DatasetChoice, DatasetPart, get_dataset
from hylfm.metrics import MetricGroup
from hylfm.run.eval_run import EvalRun
from hylfm.run.run_logger import WandbLogger
from hylfm.sampler import NoCrossBatchSampler
from hylfm.transform_pipelines import get_transforms_pipeline

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import typer

logger = logging.getLogger(__name__)


app = typer.Typer()


@app.command(name="test")
def tst(
    checkpoint: Path,
    dataset: Optional[DatasetChoice] = None,
    batch_size: int = typer.Option(1, "--batch_size"),
    data_range: float = typer.Option(1, "--data_range"),
    dataset_part: DatasetPart = typer.Option(DatasetPart.test, "--dataset_part"),
    interpolation_order: int = typer.Option(2, "--interpolation_order"),
    win_sigma: float = typer.Option(1.5, "--win_sigma"),
    win_size: int = typer.Option(11, "--win_size"),
    light_logging: bool = typer.Option(False, "--light_logging"),
    associated_train_run_name: Optional[str] = typer.Option(None, "--associated_train_run_name"),
    ui_name: Optional[str] = typer.Option(None, "--ui_name"),
):
    if ui_name is not None:
        assert associated_train_run_name is None
    elif associated_train_run_name is not None:
        assert ui_name is None
        assert dataset is None, dataset
        assert dataset_part == DatasetPart.test, dataset_part
        assert associated_train_run_name in [p.name for p in checkpoint.parents], (
            associated_train_run_name,
            checkpoint,
        )
        ui_name = associated_train_run_name

    checkpoint = Checkpoint.load(checkpoint)
    if dataset is None:
        dataset = checkpoint.config.dataset

    config = TestConfig(
        batch_size=batch_size,
        checkpoint=checkpoint,
        data_range=data_range,
        dataset=dataset,
        dataset_part=dataset_part,
        interpolation_order=interpolation_order,
        win_sigma=win_sigma,
        win_size=win_size,
    )

    model = checkpoint.model

    import wandb

    wandb_run = wandb.init(project=f"HyLFM-test", dir=str(settings.cache_dir), config=config.as_dict(), name=ui_name)


    dataset = get_dataset(config.dataset, dataset_part, transforms_pipeline)

    dataloader = DataLoader(
        dataset=dataset,
        batch_sampler=NoCrossBatchSampler(
            dataset,
            sampler_class=SequentialSampler,
            batch_sizes=[batch_size] * len(dataset.cumulative_sizes),
            drop_last=False,
        ),
        collate_fn=get_collate(batch_transformation=transforms_pipeline.batch_preprocessing),
        num_workers=settings.num_workers_test_data_loader,
        pin_memory=settings.pin_memory,
    )



    eval_run = EvalRun(
        log_pred_vs_spim=True,
        model=model,
        dataloader=dataloader,
        batch_size=batch_size,
        batch_preprocessing_in_step=transforms_pipeline.batch_preprocessing_in_step,
        batch_postprocessing=transforms_pipeline.batch_postprocessing,
        batch_premetric_trf=transforms_pipeline.batch_premetric_trf,
        metrics=metric_group,
        tgt_name=transforms_pipeline.tgt_name,
        run_logger=WandbLogger(
            point_cloud_threshold=0.3, zyx_scaling=(5, 0.7 * 8 / checkpoint.scale, 0.7 * 8 / checkpoint.scale)
        ),
        save_pred_to_disk=None if light_logging else settings.log_dir / "output_tensors" / wandb_run.name / "pred",
        save_spim_to_disk=None
        if light_logging or dataset == DatasetChoice.heart_static_fish2_f4
        else settings.log_dir / "output_tensors" / wandb_run.name / "spim",
    )

    eval_run.run()


if __name__ == "__main__":
    app()
