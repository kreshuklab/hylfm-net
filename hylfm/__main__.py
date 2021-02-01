import logging.config
from pathlib import Path

from torch.utils.data import DataLoader, SequentialSampler


from hylfm import settings
from hylfm.datasets import get_collate
from hylfm.datasets.named import DatasetName, DatasetPart, get_dataset
from hylfm.get_model import app as app_get_model
from hylfm.load_checkpoint import load_model_from_checkpoint
from hylfm.metrics import BeadPrecisionRecall
from hylfm.metrics.base import MetricGroup
from hylfm.run.eval import EvalRun
from hylfm.run.logger import WandbSummaryLogger
from hylfm.sampler import NoCrossBatchSampler
from hylfm.train import app as app_train
from hylfm.transform_pipelines import get_transforms_pipeline

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import typer

logger = logging.getLogger(__name__)

app = typer.Typer()

app.add_typer(app_get_model, name="model")
app.add_typer(app_train, name="train")


@app.command()
def preprocess(
    dataset_name: DatasetName,
    dataset_part: DatasetPart = DatasetPart.test,
    nnum: int = 19,
    z_out: int = 49,
    scale: int = 4,
    shrink: int = 8,
    interpolation_order: int = 2,
):
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


@app.command()
def test(
    checkpoint: Path,
    dataset_name: DatasetName,
    dataset_part: DatasetPart = typer.Option(DatasetPart.test, "--dataset_part"),
    batch_size: int = 1,
    interpolation_order: int = 2,
):
    import wandb

    wandb.init(project=f"HyLFM-test")
    wandb.summary["dataset"] = dataset_name
    wandb.summary["dataset_part"] = dataset_part

    model, config = load_model_from_checkpoint(checkpoint)
    nnum = model.nnum
    z_out = model.z_out
    scale = model.get_scale()
    shrink = model.get_shrink()

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

    metrics = MetricGroup(
        BeadPrecisionRecall(
            dist_threshold=3.0,
            exclude_border=False,
            max_sigma=6.0,
            min_sigma=1.0,
            overlap=0.5,
            sigma_ratio=3.0,
            threshold=0.05,
            tgt_threshold=0.05,
            scaling=(2.0, 0.7 * 8 / scale, 0.7 * 8 / scale),
        )
    )
    run = EvalRun(
        model=model,
        dataloader=dataloader,
        batch_preprocessing_in_step=transforms_pipeline.batch_preprocessing_in_step,
        batch_postprocessing=transforms_pipeline.batch_postprocessing,
        batch_premetric_trf=transforms_pipeline.batch_premetric_trf,
        metrics=metrics,
        pred_name="pred",
        tgt_name="ls_reg" if "beads" in dataset_name.value else "ls_trf",
        log_run=WandbSummaryLogger(metrics=metrics),
    )

    for batch in run:
        pass


if __name__ == "__main__":
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(asctime)s %(name)s %(levelname)s %(message)s",  # .%(msecs)03d [%(processName)s/%(threadName)s]
                    "datefmt": "%H:%M:%S",
                }
            },
            "handlers": {
                "default": {
                    "level": "DEBUG",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stdout",
                    "formatter": "default",
                }
            },
            "loggers": {
                "": {"handlers": ["default"], "level": "INFO", "propagate": True},
                "lensletnet.datasets": {"handlers": ["default"], "level": "INFO", "propagate": False},
                "lensletnet": {"handlers": ["default"], "level": "INFO", "propagate": False},
            },
        }
    )

    app()
