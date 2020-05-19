import argparse
import logging.config
import os

from pathlib import Path

import torch

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

if __name__ == "__main__":
    try:
        os.nice(10)
    except Exception as e:
        logger.error(e)

    parser = argparse.ArgumentParser(description="lnet")
    parser.add_argument("experiment_config", type=Path)
    parser.add_argument("--cuda", metavar="CUDA_VISIBLE_DEVICES", type=str, nargs="?", const="0", default=None)
    parser.add_argument("--setup", action="store_true")

    args = parser.parse_args()
    assert args.experiment_config.exists(), args.experiment_config.absolute()

    cuda_arg = args.cuda
    cuda_env = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    print('cuda env, arg', cuda_env, cuda_arg)
    if cuda_env is None:
        if cuda_arg is None:
            if torch.cuda.device_count() != 1:
                raise ValueError("'CUDA_VISIBLE_DEVICES' not specified")
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = cuda_arg
    elif cuda_arg is not None:
        if cuda_env != cuda_arg:
            raise ValueError("env and arg values for 'CUDA_VISIBLE_DEVICES' unequal!")

    from lnet.setup import Setup
    setup = Setup.from_yaml(args.experiment_config)
    if args.setup:
        setup.setup()
    else:
        setup.run()
