import logging.config

from hylfm.datasets.named import DatasetName, DatasetPart, get_dataset
from hylfm.get_model import app as app_get_model
from hylfm.train import app as app_train
from hylfm.tst import app as app_test
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
app.add_typer(app_test, name="test")


@app.command()
def preprocess(
    dataset_name: DatasetName,
    dataset_part: DatasetPart = DatasetPart.test.value,
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
