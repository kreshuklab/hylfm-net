#!/usr/bin/env python
import argparse
import subprocess
import time
from pathlib import Path

import yaml

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("prep_path", type=Path)
    parser.add_argument("meta_path", type=Path)

    args = parser.parse_args()

    with args.prep_path.open() as f:
        raw_prep = yaml.safe_load(f)

    prep = {}
    for script_name, tags in raw_prep.items():
        script_path = Path(f"./lnet/datasets/{script_name}.py")
        print(script_path)
        print("\n".join([f"\t{t}" for t in tags]))
        assert script_path.exists(), script_path
        prep[script_path] = tags

    reply = input(f"submit (y/[n])?").strip().lower()
    if reply[:1] == "y":
        for script_path, tags in prep.items():
            for tag in tags:
                subprocess.run(["sbatch", "prep.sh", str(script_path), tag, str(args.meta_path)], check=True)
#                 subprocess.run(["./prep.sh", str(script_path), tag, str(args.meta_path)], check=True)

        time.sleep(10)

    subprocess.run(["squeue", "-u", "beuttenm"])
