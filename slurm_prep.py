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

    with args.meta_path.open() as f:
        meta = yaml.safe_load(f)

    prep = {}
    for script_name, tags in raw_prep.items():
        script_path = Path(f"./lnet/datasets/{script_name}.py")
        assert script_path.exists()
        prep[script_path] = tags

    reply = input(f"submit (y/n): {prep}").strip().lower()
    if reply[:1] == "y":
        for script_path, tag in prep.items():
            subprocess.run(["sbatch", "prep.sh", str(script_path), tag, meta], shell=True, check=True)

        time.sleep(10)

    subprocess.run(["squeue", "-u", "beuttenm"], shell=True, check=True)
