import logging.config
from pathlib import Path

from merge_args import merge_args
from torch.utils.data import DataLoader, SequentialSampler

from hylfm import settings
from hylfm.datasets import get_collate_fn
from hylfm.get_model import app as app_get_model, get_model
from hylfm.datasets.named import DatasetName, DatasetPart, get_dataset
from hylfm.load_checkpoint import load_model_from_checkpoint
from hylfm.sampler import NoCrossBatchSampler
from hylfm import transformations

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
    dataset: DatasetName, part: DatasetPart = DatasetPart.whole, scale: int = 4, z_out: int = 49, nnum: int = 19
):
    get_dataset(dataset, part, meta={"scale": scale, "z_out": z_out, "nnum": nnum})


@app.command()
@merge_args(get_model)
def train(**model_kwargs,):
    config = {"model": model_kwargs}

    print(config)
    model = get_model(**model_kwargs)
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
def test(checkpoint: Path, dataset: DatasetName):
    model = load_model_from_checkpoint(checkpoint)


@app.command()
def predict(checkpoint: Path, dataset_name: DatasetName, part: DatasetPart, batch_size: int = 1):
    model, config = load_model_from_checkpoint(checkpoint)
    nnum = model.nnum()
    z_out = model.z_out
    scale = model.get_scale()
    assert nnum == config["nnum"]
    assert z_out == config["z_out"]
    assert scale == config["scale"]

    dataset = get_dataset(dataset_name, part, meta={"scale": scale, "z_out": z_out, "nnum": nnum})

    sample_augmentation =
    batch_preprocessing = transformations.Identity()
    batch_preprocessing_in_step = transformations.Cast(apply_to=["lfc", "ls_trf"], dtype="float32", device="cuda", non_blocking=True)

    dataloader = DataLoader(
        dataset=dataset,
        batch_sampler=NoCrossBatchSampler(dataset, sampler_class=SequentialSampler, batch_sizes=[batch_size] * len(dataset.cumulative_sizes), drop_last=False),
        collate_fn=get_collate_fn(batch_transformation=batch_preprocessing),
        num_workers=settings.num_workers_train_data_loader,
        pin_memory=settings.pin_memory,

        dataset=dataset,
        batch_size=batch_size,
        collate_fn=partial(collate_and_batch_trf, trf=batch_preprocessing),
        pin_memory=settings.PIN_MEMORY,
        num_workers=settings.NUM_WORKERS_TEST_DATA_LOADER,
    )



if __name__ == "__main__":
    app()
