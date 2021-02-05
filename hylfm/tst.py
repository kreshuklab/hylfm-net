from hylfm import settings  # noqa: first line to set numpy env vars

import logging.config
from pathlib import Path

from torch.utils.data import DataLoader, SequentialSampler

from hylfm import metrics, settings
from hylfm.datasets import get_collate
from hylfm.datasets.named import DatasetName, DatasetPart, get_dataset
from hylfm.load_checkpoint import load_model_from_checkpoint
from hylfm.metrics import MetricGroup
from hylfm.run.eval import EvalRun
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
    dataset_name: DatasetName,
    checkpoint: Path,
    batch_size: int = 1,
    data_range: float = 1,
    dataset_part: DatasetPart = typer.Option(DatasetPart.test.value, "--dataset_part"),
    interpolation_order: int = 2,
    win_sigma: float = 1.5,
    win_size: int = 11,
):
    import wandb

    model, config = load_model_from_checkpoint(checkpoint)
    nnum = model.nnum
    z_out = model.z_out
    scale = model.get_scale()
    shrink = model.get_shrink()
    config.update(
        dict(
            batch_size=batch_size,
            data_range=data_range,
            dataset=dataset_name.value,
            dataset_part=dataset_part.value,
            interpolation_order=interpolation_order,
            win_sigma=win_sigma,
            win_size=win_size,
            scale=scale,
            shrink=shrink,
        )
    )

    wandb_run = wandb.init(project=f"HyLFM-test", dir=str(settings.cache_dir), config=config)
    config = wandb_run.config

    transforms_pipeline = get_transforms_pipeline(
        dataset_name=dataset_name,
        dataset_part=dataset_part,
        nnum=nnum,
        z_out=z_out,
        scale=scale,
        shrink=shrink,
        interpolation_order=interpolation_order,
    )
    dataset = get_dataset(dataset_name, dataset_part, transforms_pipeline)

    dataloader = DataLoader(
        dataset=dataset,
        batch_sampler=NoCrossBatchSampler(
            dataset,
            sampler_class=SequentialSampler,
            batch_sizes=[batch_size] * len(dataset.cumulative_sizes),
            drop_last=False,
        ),
        collate_fn=get_collate(batch_transformation=transforms_pipeline.batch_preprocessing),
        num_workers=settings.num_workers_train_data_loader,
        pin_memory=settings.pin_memory,
    )

    metric_group = MetricGroup(
        # on volume
        metrics.BeadPrecisionRecall(
            dist_threshold=3.0,
            exclude_border=False,
            max_sigma=6.0,
            min_sigma=1.0,
            overlap=0.5,
            sigma_ratio=3.0,
            threshold=0.05,
            tgt_threshold=0.05,
            scaling=(2.0, 0.7 * 8 / scale, 0.7 * 8 / scale),
        ),
        metrics.MSE(),
        metrics.MS_SSIM(
            channel=1, data_range=data_range, size_average=True, spatial_dims=3, win_size=win_size, win_sigma=win_sigma
        ),
        metrics.NRMSE(),
        metrics.PSNR(data_range=data_range),
        metrics.SSIM(
            data_range=data_range, size_average=True, win_size=win_size, win_sigma=win_sigma, channel=1, spatial_dims=3
        ),
        metrics.SmoothL1(),
        # along z
        metrics.MSE(along_dim=1),
        metrics.MS_SSIM(
            along_dim=1,
            data_range=data_range,
            size_average=True,
            win_size=win_size,
            win_sigma=win_sigma,
            channel=1,
            spatial_dims=2,
        ),
        metrics.NRMSE(along_dim=1),
        metrics.PSNR(along_dim=1, data_range=data_range),
        metrics.SSIM(
            along_dim=1,
            data_range=data_range,
            size_average=True,
            win_size=win_size,
            win_sigma=win_sigma,
            channel=1,
            spatial_dims=2,
        ),
        metrics.SmoothL1(along_dim=1),
    )
    eval_run = EvalRun(
        log_pred_vs_spim=True,
        model=model,
        dataloader=dataloader,
        batch_preprocessing_in_step=transforms_pipeline.batch_preprocessing_in_step,
        batch_postprocessing=transforms_pipeline.batch_postprocessing,
        batch_premetric_trf=transforms_pipeline.batch_premetric_trf,
        metrics=metric_group,
        pred_name="pred",
        tgt_name="ls_reg" if "beads" in dataset_name.value else "ls_trf",
        run_logger=WandbLogger(point_cloud_threshold=0.2, zyx_scaling=(2, 0.7 * 8 / scale, 0.7 * 8 / scale)),
        save_pred_to_disk=settings.log_dir / "output_tensors" / wandb_run.name / "pred",
        save_spim_to_disk=settings.log_dir / "output_tensors" / wandb_run.name / "spim",
    )

    eval_run.run()


if __name__ == "__main__":
    app()
