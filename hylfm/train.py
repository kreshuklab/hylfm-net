import hylfm

from pathlib import Path
from typing import Optional

import numpy
import torch.optim
import typer
import wandb
from merge_args import merge_args
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

import hylfm.criteria
from hylfm import metrics, settings  # noqa: first line to set numpy env vars
from hylfm.checkpoint import Checkpoint, Config, TestConfig
from hylfm.datasets import get_collate
from hylfm.datasets.named import get_dataset
from hylfm.get_model import get_model
from hylfm.hylfm_types import CriterionChoice, DatasetChoice, DatasetPart, OptimizerChoice
from hylfm.metrics import MetricGroup
from hylfm.run.eval_run import EvalRun, ValidationRun
from hylfm.run.run_logger import WandbLogger, WandbValidationLogger
from hylfm.run.train_run import TrainRun
from hylfm.sampler import NoCrossBatchSampler
from hylfm.transform_pipelines import get_transforms_pipeline
from hylfm.utils.general import PeriodUnit

app = typer.Typer()


@app.command()
@merge_args(get_model)
def train(
    dataset: DatasetChoice,
    batch_multiplier: int = typer.Option(1, "--batch_multiplier"),
    batch_size: int = typer.Option(1, "--batch_size"),
    eval_batch_size: int = typer.Option(1, "--eval_batch_size"),
    criterion: CriterionChoice = CriterionChoice.WeightedSmoothL1,
    criterion_beta: float = 1.0,
    criterion_threshold: float = 1.0,
    criterion_weight: float = 0.05,
    criterion_apply_below_threshold: bool = True,
    data_range: float = typer.Option(1.0, "--data_range"),
    interpolation_order: int = 2,
    lr: float = 1e-4,
    max_epochs: int = typer.Option(10, "--max_epochs"),
    model_weights: Optional[Path] = typer.Option(None, "--model_weights"),
    optimizer: OptimizerChoice = OptimizerChoice.Adam,
    patience: int = typer.Option(5, "--patience"),
    seed: Optional[int] = None,
    validate_every_unit: PeriodUnit = typer.Option(PeriodUnit.epoch, "--validate_every_unit"),
    validate_every_value: int = typer.Option(1, "--validate_every_value"),
    weight_decay: float = typer.Option(0.0, "--weight_decay"),
    win_sigma: float = typer.Option(1.5, "--win_sigma"),
    win_size: int = typer.Option(11, "--win_size"),
    **model_kwargs,
):
    config = Config(
        batch_multiplier=batch_multiplier,
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        criterion=criterion,
        criterion_beta=criterion_beta,
        criterion_threshold=criterion_threshold,
        criterion_weight=criterion_weight,
        criterion_apply_below_threshold=criterion_apply_below_threshold,
        data_range=data_range,
        dataset=dataset,
        interpolation_order=interpolation_order,
        lr=lr,
        max_epochs=max_epochs,
        model=model_kwargs,
        model_weights=model_weights,
        optimizer=optimizer,
        patience=patience,
        seed=seed,
        validate_every_unit=validate_every_unit,
        validate_every_value=validate_every_value,
        weight_decay=weight_decay,
        win_sigma=win_sigma,
        win_size=win_size,
    )

    wandb_run = wandb.init(
        project="HyLFM-train",
        dir=str(settings.cache_dir),
        config=config.as_dict(for_logging=True),
        reinit=False,
        resume="never",
    )

    if model_weights is not None:
        model_weights = Checkpoint.load(model_weights).model_weights

    checkpoint = Checkpoint(
        model_weights=model_weights, config=config, training_run_name=wandb_run.name, training_run_id=wandb_run.id
    )

    train_from_checkpoint(wandb_run, checkpoint=checkpoint)


def resume(checkpoint: Path):
    checkpoint = Checkpoint.load(checkpoint)
    wandb_run = wandb.init(
        project="HyLFM-train",
        dir=str(settings.cache_dir),
        config=checkpoint.config.as_dict(checkpoint.model),
        reinit=False,
        resume="must",
        name=checkpoint.training_run_name,
        id=checkpoint.training_run_id,
    )
    train_from_checkpoint(wandb_run, checkpoint)


def train_from_checkpoint(wandb_run, checkpoint: Checkpoint):
    cfg = checkpoint.config

    if cfg.seed is not None:
        numpy.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

    model = checkpoint.model

    nnum = model.nnum
    z_out = model.z_out
    scale = checkpoint.scale

    # todo: get from checkpoint like model and restore optimizer
    opt: torch.optim.Optimizer = getattr(torch.optim, cfg.optimizer.name)(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    opt.zero_grad()

    if cfg.criterion == CriterionChoice.MS_SSIM:
        criterion_kwargs = dict(
            channel=1,
            data_range=cfg.data_range,
            size_average=True,
            spatial_dims=3,
            win_size=cfg.win_size,
            win_sigma=cfg.win_sigma,
        )
    elif cfg.criterion == CriterionChoice.SmoothL1:
        criterion_kwargs = dict(beta=cfg.criterion_beta)
    elif cfg.criterion == CriterionChoice.SmoothL1_MS_SSIM:
        criterion_kwargs = dict(
            beta=cfg.criterion_beta,
            channel=1,
            data_range=cfg.data_range,
            size_average=True,
            spatial_dims=3,
            win_size=cfg.win_size,
            win_sigma=cfg.win_sigma,
            ms_ssim_weight=0.001,
        )
    elif cfg.criterion == CriterionChoice.WeightedSmoothL1:
        criterion_kwargs = dict(
            threshold=cfg.criterion_threshold,
            weight=cfg.criterion_weight,
            apply_below_threshold=cfg.criterion_apply_below_threshold,
            beta=cfg.criterion_beta,
        )
    else:
        raise NotImplementedError(cfg.criterion)

    crit = getattr(hylfm.criteria, cfg.criterion)(**criterion_kwargs)

    transforms_pipelines = {
        part: get_transforms_pipeline(
            dataset_name=cfg.dataset,
            dataset_part=part,
            nnum=nnum,
            z_out=z_out,
            scale=scale,
            shrink=checkpoint.shrink,
            interpolation_order=cfg.interpolation_order,
        )
        for part in DatasetPart
    }

    datasets = {part: get_dataset(cfg.dataset, part, transforms_pipelines[part]) for part in DatasetPart}
    dataloaders = {
        part: DataLoader(
            dataset=datasets[part],
            batch_sampler=NoCrossBatchSampler(
                datasets[part],
                sampler_class=RandomSampler if part == DatasetPart.train else SequentialSampler,
                batch_sizes=[cfg.batch_size if part == DatasetPart.train else cfg.eval_batch_size]
                * len(datasets[part].cumulative_sizes),
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
            # metrics.BeadPrecisionRecall(
            #     dist_threshold=3.0,
            #     exclude_border=False,
            #     max_sigma=6.0,
            #     min_sigma=1.0,
            #     overlap=0.5,
            #     sigma_ratio=3.0,
            #     threshold=0.05,
            #     tgt_threshold=0.05,
            #     scaling=(2.0, 0.7 * 8 / scale, 0.7 * 8 / scale),
            # ),
            metrics.MSE(),
            metrics.MS_SSIM(
                channel=1,
                data_range=cfg.data_range,
                size_average=True,
                spatial_dims=3,
                win_size=cfg.win_size,
                win_sigma=cfg.win_sigma,
            ),
            metrics.NRMSE(),
            metrics.PSNR(data_range=cfg.data_range),
            metrics.SSIM(
                data_range=cfg.data_range,
                size_average=True,
                win_size=cfg.win_size,
                win_sigma=cfg.win_sigma,
                channel=1,
                spatial_dims=3,
            ),
            metrics.SmoothL1(),
        ),
        DatasetPart.test: MetricGroup(
            # on volume
            # metrics.BeadPrecisionRecall(
            #     dist_threshold=3.0,
            #     exclude_border=False,
            #     max_sigma=6.0,
            #     min_sigma=1.0,
            #     overlap=0.5,
            #     sigma_ratio=3.0,
            #     threshold=0.05,
            #     tgt_threshold=0.05,
            #     scaling=(2.0, 0.7 * 8 / scale, 0.7 * 8 / scale),
            # ),
            metrics.MSE(),
            metrics.MS_SSIM(
                channel=1,
                data_range=cfg.data_range,
                size_average=True,
                spatial_dims=3,
                win_size=cfg.win_size,
                win_sigma=cfg.win_sigma,
            ),
            metrics.NRMSE(),
            metrics.PSNR(data_range=cfg.data_range),
            metrics.SSIM(
                data_range=cfg.data_range,
                size_average=True,
                win_size=cfg.win_size,
                win_sigma=cfg.win_sigma,
                channel=1,
                spatial_dims=3,
            ),
            metrics.SmoothL1(),
            # along z
            metrics.MSE(along_dim=1),
            metrics.MS_SSIM(
                along_dim=1,
                data_range=cfg.data_range,
                size_average=True,
                win_size=cfg.win_size,
                win_sigma=cfg.win_sigma,
                channel=1,
                spatial_dims=2,
            ),
            metrics.NRMSE(along_dim=1),
            metrics.PSNR(along_dim=1, data_range=cfg.data_range),
            metrics.SSIM(
                along_dim=1,
                data_range=cfg.data_range,
                size_average=True,
                win_size=cfg.win_size,
                win_sigma=cfg.win_sigma,
                channel=1,
                spatial_dims=2,
            ),
            metrics.SmoothL1(along_dim=1),
        ),
    }

    part = DatasetPart.validate
    score_metric = "MS-SSIM"
    validator = ValidationRun(
        batch_postprocessing=transforms_pipelines[part].batch_postprocessing,
        batch_premetric_trf=transforms_pipelines[part].batch_premetric_trf,
        batch_preprocessing_in_step=transforms_pipelines[part].batch_preprocessing_in_step,
        dataloader=dataloaders[part],
        batch_size=cfg.eval_batch_size,
        log_pred_vs_spim=False,
        metrics=metric_groups[part],
        minimize=getattr(metrics, score_metric.replace("-", "_")).minimize,
        model=model,
        pred_name="pred",
        run_logger=WandbValidationLogger(
            point_cloud_threshold=0.3,
            zyx_scaling=(2, 0.7 * 8 / scale, 0.7 * 8 / scale),
            score_metric=score_metric,
            minimize=getattr(metrics, score_metric.replace("-", "_")).minimize,
        ),
        save_pred_to_disk=None,
        save_spim_to_disk=None,
        score_metric=score_metric,
        tgt_name="ls_reg" if "beads" in cfg.dataset.value else "ls_trf",
    )

    part = DatasetPart.train
    train_run = TrainRun(
        batch_postprocessing=transforms_pipelines[part].batch_postprocessing,
        batch_premetric_trf=transforms_pipelines[part].batch_premetric_trf,
        batch_preprocessing_in_step=transforms_pipelines[part].batch_preprocessing_in_step,
        criterion=crit,
        dataloader=dataloaders[part],
        batch_size=cfg.batch_size,
        metrics=metric_groups[part],
        model=model,
        optimizer=opt,
        pred_name="pred",
        run_logger=WandbLogger(point_cloud_threshold=0.3, zyx_scaling=(2, 0.7 * 8 / scale, 0.7 * 8 / scale)),
        tgt_name="ls_reg" if "beads" in cfg.dataset.value else "ls_trf",
        train_metrics=metric_groups[part],
        validator=validator,
        checkpoint=checkpoint,
    )

    # part = DatasetPart.test
    # tester = EvalRun(
    #     batch_postprocessing=transforms_pipelines[part].batch_postprocessing,
    #     batch_premetric_trf=transforms_pipelines[part].batch_premetric_trf,
    #     batch_preprocessing_in_step=transforms_pipelines[part].batch_preprocessing_in_step,
    #     dataloader=dataloaders[part],
    #     batch_size=cfg.eval_batch_size,
    #     log_pred_vs_spim=False,
    #     metrics=metric_groups[part],
    #     model=model,
    #     pred_name="pred",
    #     run_logger=WandbLogger(point_cloud_threshold=0.3, zyx_scaling=(2, 0.7 * 8 / scale, 0.7 * 8 / scale)),
    #     save_pred_to_disk=None,
    #     save_spim_to_disk=None,
    #     tgt_name="ls_reg" if "beads" in cfg.dataset.value else "ls_trf",
    # )
    #
    # test_config = TestConfig(
    #     batch_size=cfg.eval_batch_size,
    #     checkpoint=checkpoint,
    #     data_range=cfg.data_range,
    #     dataset=cfg.dataset,
    #     dataset_part=DatasetPart.test,
    #     interpolation_order=cfg.interpolation_order,
    #     win_sigma=cfg.win_sigma,
    #     win_size=cfg.win_size,
    # )

    # actually train
    train_run.fit()
    wandb_run.finish()

    # todo: start test in separate process
    # # run default test
    # wandb_run = wandb.init(
    #     name=wandb_run.name,
    #     project=f"HyLFM-test",
    #     dir=str(settings.cache_dir),
    #     config=test_config.as_dict(),
    #     reinit=True,
    # )
    # tester.run()
    # wandb_run.finish()


if __name__ == "__main__":
    app()
