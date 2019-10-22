import argparse
import logging.config
import os

from pathlib import Path


CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s.%(msecs)03d [%(processName)s/%(threadName)s] %(levelname)s %(message)s",
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
    parser = argparse.ArgumentParser(description="lnet")
    parser.add_argument("experiment_config", type=Path)
    parser.add_argument("CUDA_VISIBLE_DEVICES", type=str)
    parser.add_argument("--test", action="store_true")

    args = parser.parse_args()
    assert args.experiment_config.exists(), args.experiment_config.absolute()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.CUDA_VISIBLE_DEVICES

    from lnet.experiment import Experiment

    exp = Experiment(config_path=args.experiment_config)
    if args.test:
        exp.test()
    else:
        exp.run()
