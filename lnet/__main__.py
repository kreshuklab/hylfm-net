import argparse

from pathlib import Path

from lnet.experiment import Experiment


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="lnet")
    parser.add_argument("experiment_config", type=Path)
    parser.add_argument("--test", action="store_true")

    args = parser.parse_args()

    exp = Experiment(config_path=args.experiment_config)
    if args.test:
        exp.test()
    else:
        exp.run()
