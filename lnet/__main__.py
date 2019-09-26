import argparse

from pathlib import Path

from lnet.experiment import runnable_experiments


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="lnet")
    parser.add_argument("config", type=Path)

    args = parser.parse_args()

    for Exp in runnable_experiments():
        Exp(config=args.config).run()

