import argparse
import datetime
import logging.config
from pathlib import Path
from subprocess import STDOUT, TimeoutExpired, check_output
from time import sleep

from lnet.datasets.base import GKRESHUK

LOG_DIR = Path(GKRESHUK) / "beuttenm/repos/lnet/logs"

CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s %(name)s %(levelname)s %(message)s",  # .%(msecs)03d [%(processName)s/%(threadName)s]
            "datefmt": "%Y-%m-%d %H:%M:%S",
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
    "loggers": {"": {"handlers": ["default"], "level": "INFO", "propagate": True}},
}
logging.config.dictConfig(CONFIG)
logger = logging.getLogger(__name__)


def run_tensorboard(*, logdir: str, start: datetime.time, stop: datetime.time):
    now = datetime.datetime.now()
    start = datetime.datetime.combine(now.date(), start)
    if start < now:
        start += datetime.timedelta(days=1)

    stop = datetime.datetime.combine(now.date(), stop)
    if stop < now:
        stop += datetime.timedelta(days=1)

    assert start > now, start
    assert stop > now, stop
    while True:
        if start.weekday() > 5:
            start + datetime.timedelta(days=1)

        duration = (min(start, stop) - datetime.datetime.now()).total_seconds()
        if stop < start:
            logger.info("running for %ds until %s", duration, stop)
            try:
                check_output(
                    ["tensorboard", "--logdir", str(logdir), "--port", "5158"], stderr=STDOUT, timeout=duration
                )
            except TimeoutExpired:
                pass

            stop += datetime.timedelta(days=1)
        else:
            logger.info("sleeping for %ds until %s", duration, start)
            sleep(duration)
            start += datetime.timedelta(days=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("logdir", type=str)
    parser.add_argument("--start", nargs="+", type=int, default=(7,))
    parser.add_argument("--stop", nargs="+", type=int, default=(20,))

    args = parser.parse_args()
    start = datetime.time(*args.start)
    stop = datetime.time(*args.stop)
    run_tensorboard(logdir=LOG_DIR / args.logdir, start=start, stop=stop)
