import argparse
import datetime
from pathlib import Path
from subprocess import STDOUT, TimeoutExpired, check_output
from time import sleep

LOG_DIR = Path("/g/kreshuk/beuttenm/repos/lnet/logs")


def run_tensorboard(*, logdir: str, start: datetime.time, stop: datetime.time):
    now = datetime.datetime.now()
    start = datetime.datetime.combine(now.date(), start)
    if start < now:
        start + datetime.timedelta(days=1)

    stop = datetime.datetime.combine(now.date(), stop)
    if stop < now:
        stop + datetime.timedelta(days=1)

    while True:
        if start.weekday() > 5:
            start + datetime.timedelta(days=1)
        elif start < stop:
            print("running until", stop)
            try:
                check_output(
                    ["tensorboard", "--logdir", str(logdir)], stderr=STDOUT, timeout=(stop - start).total_seconds()
                )
            except TimeoutExpired:
                pass

            start + datetime.timedelta(days=1)
        else:
            print("sleeping until", start)
            sleep((start - stop).total_seconds())
            stop + datetime.timedelta(days=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("logdir", type=str)
    parser.add_argument("--start", nargs="+", type=int, default=(7,))
    parser.add_argument("--stop", nargs="+", type=int, default=(20,))

    args = parser.parse_args()
    start = datetime.time(*args.start)
    stop = datetime.time(*args.stop)
    run_tensorboard(logdir=LOG_DIR / args.logdir, start=start, stop=stop)
