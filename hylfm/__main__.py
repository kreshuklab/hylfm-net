import argparse
import logging.config
import os
import shutil
import sys
import warnings
from pathlib import Path

from hylfm import settings
from hylfm.utils.general import download_file_from_zenodo

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


def main(args=None):

    if args is None:
        parser = argparse.ArgumentParser(description="lnet")
        parser.add_argument("experiment_config", type=Path)
        parser.add_argument("--cuda", metavar="CUDA_VISIBLE_DEVICES", type=str, nargs="?", default=None)
        parser.add_argument("--setup", action="store_true")
        parser.add_argument("--checkpoint", default=None)

        args = parser.parse_args()

    experiment_config: Path = args.experiment_config
    assert experiment_config.exists(), experiment_config.absolute()

    checkpoint = args.checkpoint
    if checkpoint == "small_beads_demo":
        small_beads_demo_doi = "10.5281/zenodo.4036556"
        small_beads_demo_file_name = "small_beads_v1_weights_SmoothL1Loss%3D-0.00012947025970788673.pth"
        checkpoint = settings.download_dir / small_beads_demo_doi / small_beads_demo_file_name
        download_file_from_zenodo(small_beads_demo_doi, small_beads_demo_file_name, checkpoint)
    elif checkpoint is not None:
        checkpoint = Path(checkpoint)

    cuda_arg = args.cuda
    cuda_env = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    logger.info("cuda env: %s, cuda arg: %s", cuda_env, cuda_arg)
    if cuda_env is None:
        if cuda_arg is None:
            # import torch
            #
            # if torch.cuda.device_count() != 1:
            #     raise ValueError("'CUDA_VISIBLE_DEVICES' not specified")
            warnings.warn("'CUDA_VISIBLE_DEVICES' not specified")
            cuda_arg = ""

        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_arg
    elif cuda_arg is None:
        cuda_arg = cuda_env
    else:
        if cuda_env != cuda_arg:
            raise ValueError("env and arg values for 'CUDA_VISIBLE_DEVICES' unequal!")

    device = "cpu" if not cuda_arg else 0

    from hylfm.setup import Setup

    setup = Setup.from_yaml(experiment_config, checkpoint=checkpoint, device=device)

    if args.setup:
        log_path = setup.setup()
        shutil.rmtree(log_path)
    else:
        logger.info(
            "run tensorboard with:\ntensorboard --logdir=%s\nor:\ntensorboard --logdir=%s",
            settings.log_dir,
            setup.log_path.parent,
        )
        log_path = setup.run()
        logger.info("done logging to %s", log_path)


if __name__ == "__main__":
    sys.exit(main())
