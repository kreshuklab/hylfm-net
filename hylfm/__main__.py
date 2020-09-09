import argparse
import logging.config
import os
import shutil
import sys
from pathlib import Path
from typing import Optional

from hylfm import settings

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
        parser.add_argument("--cuda", metavar="CUDA_VISIBLE_DEVICES", type=str, nargs="?", const="0", default=None)
        parser.add_argument("--setup", action="store_true")
        parser.add_argument("--checkpoint", type=Path, default=None)
        parser.add_argument("--test", action="store_true")
        parser.add_argument("--delete_existing_log_folder", action="store_true")

        args = parser.parse_args()

    experiment_config: Path = args.experiment_config
    assert experiment_config.exists(), experiment_config.absolute()

    checkpoint: Optional[Path] = args.checkpoint
    if args.test and checkpoint is None:
        raise TypeError("cannot test without checkpoint")

    cuda_arg = args.cuda
    cuda_env = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    logger.info("cuda env: %s, arg: %s", cuda_env, cuda_arg)
    if cuda_env is None:
        if cuda_arg is None:
            if torch.cuda.device_count() != 1:
                raise ValueError("'CUDA_VISIBLE_DEVICES' not specified")
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = cuda_arg
    elif cuda_arg is not None:
        if cuda_env != cuda_arg:
            raise ValueError("env and arg values for 'CUDA_VISIBLE_DEVICES' unequal!")

    from hylfm.setup import Setup

    setup = Setup.from_yaml(experiment_config, checkpoint=checkpoint)

    if args.setup:
        log_path = setup.setup()
        shutil.rmtree(log_path)
    else:
        print(f"run tensorboard with:\ntensorboard --logdir={settings.log_dir}")
        print(f"or:\ntensorboard --logdir={setup.log_path.parent}")
        log_path = setup.run()
        print(f"done logging to {log_path}")


if __name__ == "__main__":
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    try:
        os.nice(10)
    except Exception as e:
        logger.error(e)

    import torch.multiprocessing

    if settings.multiprocessing_start_method:
        torch.multiprocessing.set_start_method(settings.multiprocessing_start_method)

    sys.exit(main())
