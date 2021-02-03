import hylfm  # noqa

import torch.optim
import typer
import wandb
from merge_args import merge_args


import hylfm.criteria
from hylfm.get_model import get_model
from hylfm.hylfm_types import DatasetName, DatasetPart
from hylfm.metrics import MetricGroup
from hylfm.run.eval import EvalRun
from hylfm.run.train import TrainRun
from hylfm.transform_pipelines import get_transforms_pipeline

app = typer.Typer()


@app.command()
@merge_args(get_model)
def train(
    dataset_name: DatasetName = typer.Option(..., "--dataset_name"),
    interpolation_order: int = 2,
    batch_size: int = typer.Option(1, "--batch_size"),
    batch_multiplier: int = typer.Option(1, "--batch_multiplier"),
    max_epochs: int = 10,
    optimizer: str = "Adam",
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    loss: str = "MS_SSIM",
    **model_kwargs,
):
    print([p for p in DatasetPart])
    assert False
    model = get_model(**model_kwargs)
    opt: torch.optim.Optimizer = getattr(torch.optim, optimizer)(
        parameters=model.parameters(), lr=lr, weight_decay=weight_decay
    )
    criterion = getattr(hylfm.criteria, loss)

    wandb.init(project=f"HyLFM-train")
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
    data_range = 1
    win_sigma = 1.5
    win_size = 11

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
    validator = EvalRun(
        log_pred_vs_spim=False,
        model=model,
        dataloader=dataloader,
        batch_preprocessing_in_step=transforms_pipeline.batch_preprocessing_in_step,
        batch_postprocessing=transforms_pipeline.batch_postprocessing,
        batch_premetric_trf=transforms_pipeline.batch_premetric_trf,
        metrics=metric_group,
        pred_name="pred",
        tgt_name="ls_reg" if "beads" in dataset_name.value else "ls_trf",
        run_logger=WandbLogger(zyx_scaling=(2, 0.7 * 8 / scale, 0.7 * 8 / scale)),
    )

    validator = EvalRun(model=model)
    train_metrics = MetricGroup()
    run = TrainRun(
        max_epoch=max_epochs,
        train_metric=train_metrics,
        validato=validator,
        criterio=criterion,
        batch_multiplie=batch_multiplier,
        optimizer=opt,
        model=model,
    )


if __name__ == "__main__":
    train()
