from inspect import signature
import logging.config
from pathlib import Path

from merge_args import merge_args
from torch.utils.data import DataLoader, SequentialSampler

from hylfm import settings
from hylfm.datasets import get_collate
from hylfm.datasets.named import DatasetName, DatasetPart, get_dataset
from hylfm.get_model import app as app_get_model, get_model
from hylfm.hylfm_types import TransformsPipeline
from hylfm.load_checkpoint import load_model_from_checkpoint
from hylfm.run.predict import PredictRun
from hylfm.sampler import NoCrossBatchSampler
from hylfm.transform_pipelines import get_transforms_pipeline
from hylfm.utils.general import return_unused_kwargs_to

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import typer

CONFIG = {
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
logging.config.dictConfig(CONFIG)
logger = logging.getLogger(__name__)

app = typer.Typer()

app.add_typer(app_get_model, name="model")


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
@merge_args(get_model)
def train(**model_kwargs):
    print(model_kwargs)
    config = {"model": model_kwargs}

    print(config)
    # model = get_model(**model_kwargs)
    # if checkpoint == "small_beads_demo":
    #     small_beads_demo_doi = "10.5281/zenodo.4036556"
    #     small_beads_demo_file_name = "small_beads_v1_weights_SmoothL1Loss%3D-0.00012947025970788673.pth"
    #     checkpoint = settings.download_dir / small_beads_demo_doi / small_beads_demo_file_name
    #     download_file_from_zenodo(small_beads_demo_doi, small_beads_demo_file_name, checkpoint)
    # elif checkpoint is not None:
    #     checkpoint = Path(checkpoint)

    # from hylfm.setup import Setup
    #
    # setup = Setup.from_yaml(experiment_config)
    #
    # if args.setup:
    #     log_path = setup.setup()
    #     shutil.rmtree(log_path)
    # else:
    #     logger.info(
    #         "run tensorboard with:\ntensorboard --logdir=%s\nor:\ntensorboard --logdir=%s",
    #         settings.log_dir,
    #         setup.log_path.parent,
    #     )
    #     log_path = setup.run()
    #     logger.info("done logging to %s", log_path)


@app.command()
def test(checkpoint: Path, dataset: DatasetName, part: DatasetPart = DatasetPart.test):
    model, config = load_model_from_checkpoint(checkpoint)


@app.command()
def predict(
    checkpoint: Path,
    dataset_name: DatasetName,
    dataset_part: DatasetPart,
    batch_size: int = 1,
    interpolation_order: int = 2,
):
    import wandb

    wandb.init(project=f"predict-{dataset_name}-{dataset_part}")
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

    run = PredictRun(
        model=model,
        dataloader=dataloader,
        batch_preprocessing_in_step=transforms_pipeline.batch_preprocessing_in_step,
        batch_postprocessing=transforms_pipeline.batch_postprocessing,
    )

    for batch in run:
        print(batch)


if __name__ == "__main__":
    app()
