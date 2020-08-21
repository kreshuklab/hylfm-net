import logging
import subprocess
from argparse import ArgumentParser
from math import ceil
from pathlib import Path
from random import shuffle

from ruamel.yaml import YAML

from plain.plain_setup import get_setup

logger = logging.getLogger(__name__)

yaml = YAML(typ="safe")


if __name__ == "__main__":
    parser = ArgumentParser(description="care inference")
    parser.add_argument("cuda_visible_devices")
    parser.add_argument("subpath", default="heart/dynamic/v0_spe1000_on_48x88x88")
    args = parser.parse_args()

    setup = get_setup(subpath=args.subpath)

    cuda_devices = args.cuda_visible_devices.split(",")
    gpus = list(range(len(cuda_devices)))
    ngpus = len(gpus)
    pools_per_gpu = 1
    npools = ngpus * pools_per_gpu
    per_pool_workers = 1

    lr_path = setup["test_data_path"] / "lr"
    print("get lr from ", lr_path)
    assert lr_path.exists(), lr_path.absolute()
    file_paths = list(map(str, lr_path.glob("*.tif")))
    shuffle(file_paths)
    chunk_size = ceil(len(file_paths) / npools)
    print("chunk_size", chunk_size)
    file_path_chunks = [file_paths[i * chunk_size : (i + 1) * chunk_size] for i in range(npools)]

    chunk_dir = Path("chunks")
    chunk_dir.mkdir(exist_ok=True)

    print("chunk_dir", chunk_dir.absolute())
    parallel_runs = []
    for pool_id in range(npools):
        gpu = pool_id % ngpus
        chunk_file_path = chunk_dir / f"{cuda_devices[gpu]}.yml"
        yaml.dump(file_path_chunks[pool_id], chunk_file_path)

        parallel_runs.append(
            subprocess.Popen(
                [
                    # " ".join([
                    "/g/kreshuk/beuttenm/miniconda3/envs/csbdeep-gpu/bin/python",
                    "inference.py",
                    cuda_devices[gpu],
                    args.subpath,
                    str(chunk_file_path),
                ]
            )
        )

    print("n prallel:", len(parallel_runs))
    [run.communicate() for run in parallel_runs]
