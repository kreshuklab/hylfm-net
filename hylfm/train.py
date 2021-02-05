from enum import Enum
from pathlib import Path
from typing import Optional

import torch.optim
import typer
import wandb
from merge_args import merge_args
from torch.utils.data import DataLoader, SequentialSampler

import hylfm
import hylfm.criteria
from hylfm import metrics, settings
from hylfm.datasets import get_collate
from hylfm.datasets.named import get_dataset
from hylfm.get_model import get_model
from hylfm.hylfm_types import DatasetName, DatasetPart
from hylfm.load_checkpoint import load_state_from_checkpoint
from hylfm.metrics import MetricGroup
from hylfm.run.eval import EvalRun, ValidationRun
from hylfm.run.run_logger import WandbLogger
from hylfm.run.train_run import TrainRun
from hylfm.sampler import NoCrossBatchSampler
from hylfm.transform_pipelines import get_transforms_pipeline
from hylfm.utils.general import Period, PeriodUnit

app = typer.Typer()


@app.command()
@merge_args(get_model)
def train(
    batch_multiplier: int = typer.Option(1, "--batch_multiplier"),
    batch_size: int = typer.Option(1, "--batch_size"),
    data_range: float = typer.Option(1.0, "--data_range"),
    dataset_name: DatasetName = typer.Option(..., "--dataset_name"),
    init_weights_from: Optional[Path] = typer.Option(None, "--init_weights_from"),
    interpolation_order: int = 2,
    loss: str = "MS_SSIM",
    lr: float = 1e-3,
    max_epochs: int = typer.Option(10, "--max_epochs"),
    optimizer: str = "Adam",
    patience: int = typer.Option(10, "--patience"),
    validate_every_unit: PeriodUnit = typer.Option(PeriodUnit.epoch.value, "--validate_every_unit"),
    validate_every_value: int = typer.Option(1, "--validate_every_value"),
    weight_decay: float = typer.Option(0.0, "--weight_decay"),
    win_sigma: float = typer.Option(1.5, "--win_sigma"),
    win_size: int = typer.Option(11, "--win_size"),
    **model_kwargs,
):
    config = {
        k: v.value if isinstance(v, Enum) else str(v) if isinstance(v, Path) else v
        for k, v in dict(
            batch_multiplier=batch_multiplier,
            batch_size=batch_size,
            data_range=data_range,
            dataset=dataset_name,
            init_weights_from=init_weights_from,
            interpolation_order=interpolation_order,
            loss=loss,
            lr=lr,
            max_epochs=max_epochs,
            model={k: v.value if isinstance(v, Enum) else v for k, v in model_kwargs.items()},
            optimizer=optimizer,
            patience=patience,
            validate_every_unit=validate_every_unit,
            validate_every_value=validate_every_value,
            weight_decay=weight_decay,
            win_sigma=win_sigma,
            win_size=win_size,
        ).items()
    }
    wandb_run = wandb.init(project=f"HyLFM-train", dir=str(settings.cache_dir), config=config, reinit=True)
    config = wandb_run.config  # overwrites config during sweep?!?

    model = get_model(**model_kwargs)
    if init_weights_from is not None:
        state = load_state_from_checkpoint(init_weights_from)
        model.load_state_dict(state["model"], strict=True)

    opt: torch.optim.Optimizer = getattr(torch.optim, optimizer)(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    criterion = getattr(hylfm.criteria, loss)

    wandb.summary["dataset"] = dataset_name

    nnum = model.nnum
    z_out = model.z_out
    scale = model.get_scale()
    shrink = model.get_shrink()

    transforms_pipelines = {
        part: get_transforms_pipeline(
            dataset_name=dataset_name,
            dataset_part=part,
            nnum=nnum,
            z_out=z_out,
            scale=scale,
            shrink=shrink,
            interpolation_order=interpolation_order,
        )
        for part in DatasetPart
    }

    datasets = {part: get_dataset(dataset_name, part, transforms_pipelines[part]) for part in DatasetPart}
    dataloaders = {
        part: DataLoader(
            dataset=datasets[part],
            batch_sampler=NoCrossBatchSampler(
                datasets[part],
                sampler_class=SequentialSampler,
                batch_sizes=[batch_size] * len(datasets[part].cumulative_sizes),
                drop_last=False,
            ),
            collate_fn=get_collate(batch_transformation=transforms_pipelines[part].batch_preprocessing),
            num_workers=settings.num_workers_train_data_loader,
            pin_memory=settings.pin_memory,
        )
        for part in DatasetPart
    }

    metric_groups = {
        DatasetPart.train: MetricGroup(),
        DatasetPart.validate: MetricGroup(
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
                channel=1,
                data_range=data_range,
                size_average=True,
                spatial_dims=3,
                win_size=win_size,
                win_sigma=win_sigma,
            ),
            metrics.NRMSE(),
            metrics.PSNR(data_range=data_range),
            metrics.SSIM(
                data_range=data_range,
                size_average=True,
                win_size=win_size,
                win_sigma=win_sigma,
                channel=1,
                spatial_dims=3,
            ),
            metrics.SmoothL1(),
        ),
        DatasetPart.test: MetricGroup(
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
                channel=1,
                data_range=data_range,
                size_average=True,
                spatial_dims=3,
                win_size=win_size,
                win_sigma=win_sigma,
            ),
            metrics.NRMSE(),
            metrics.PSNR(data_range=data_range),
            metrics.SSIM(
                data_range=data_range,
                size_average=True,
                win_size=win_size,
                win_sigma=win_sigma,
                channel=1,
                spatial_dims=3,
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
        ),
    }

    part = DatasetPart.validate
    validator = ValidationRun(
        batch_postprocessing=transforms_pipelines[part].batch_postprocessing,
        batch_premetric_trf=transforms_pipelines[part].batch_premetric_trf,
        batch_preprocessing_in_step=transforms_pipelines[part].batch_preprocessing_in_step,
        dataloader=dataloaders[part],
        log_pred_vs_spim=False,
        metrics=metric_groups[part],
        minimize=False,
        model=model,
        pred_name="pred",
        run_logger=WandbLogger(zyx_scaling=(2, 0.7 * 8 / scale, 0.7 * 8 / scale)),
        save_pred_to_disk=None,
        save_spim_to_disk=None,
        score_metric="MS-SSIM",
        tgt_name="ls_reg" if "beads" in dataset_name.value else "ls_trf",
    )

    part = DatasetPart.train
    train_run = TrainRun(
        batch_multiplie=batch_multiplier,
        batch_postprocessing=transforms_pipelines[part].batch_postprocessing,
        batch_premetric_trf=transforms_pipelines[part].batch_premetric_trf,
        batch_preprocessing_in_step=transforms_pipelines[part].batch_preprocessing_in_step,
        criterion=criterion,
        dataloader=dataloaders[part],
        max_epochs=max_epochs,
        model=model,
        optimizer=opt,
        patience=patience,
        pred_name="pred",
        run_logger=WandbLogger(zyx_scaling=(2, 0.7 * 8 / scale, 0.7 * 8 / scale)),
        tgt_name="ls_reg" if "beads" in dataset_name.value else "ls_trf",
        train_metrics=metric_groups[part],
        validate_every=Period(validate_every_value, validate_every_unit),
        validator=validator,
    )

    part = DatasetPart.test
    tester = EvalRun(
        batch_postprocessing=transforms_pipelines[part].batch_postprocessing,
        batch_premetric_trf=transforms_pipelines[part].batch_premetric_trf,
        batch_preprocessing_in_step=transforms_pipelines[part].batch_preprocessing_in_step,
        dataloader=dataloaders[part],
        log_pred_vs_spim=False,
        metrics=metric_groups[part],
        model=model,
        pred_name="pred",
        run_logger=WandbLogger(zyx_scaling=(2, 0.7 * 8 / scale, 0.7 * 8 / scale)),
        save_pred_to_disk=None,
        save_spim_to_disk=None,
        tgt_name="ls_reg" if "beads" in dataset_name.value else "ls_trf",
    )

    # train
    train_run.fit()

    # test
    wandb_run = wandb.init(
        name=wandb_run.name, project=f"HyLFM-test", dir=str(settings.cache_dir), config=config, reinit=True
    )
    tester.run()


if __name__ == "__main__":
    app()
