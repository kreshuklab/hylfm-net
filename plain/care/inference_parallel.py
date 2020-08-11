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
    # else:
    #     args = None
    #
    # from tensorflow.python.client import device_lib
    #
    #
    # def get_available_gpus():
    #     local_device_protos = device_lib.list_local_devices()
    #     return [x.name for x in local_device_protos if x.device_type == "GPU"]
    #
    #
    # if __name__ == "__main__":
    setup = get_setup(subpath=args.subpath)
    # available_gpus = get_available_gpus()
    # print("available_gpus", available_gpus)

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
            subprocess.Popen([
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
    # for pr in parallel_runs:
    #     print(pr)

# import os
# from argparse import ArgumentParser
# from concurrent.futures import ThreadPoolExecutor, as_completed
# from queue import Queue, SimpleQueue
#
# from tifffile import imread
# from tqdm import tqdm
#
# if __name__ == "__main__":
#     parser = ArgumentParser(description="care inference")
#     parser.add_argument("cuda_visible_devices")
#     parser.add_argument("subpath", default="heart/dynamic")
#     parser.add_argument("model_name", default="v0_spe1000_on_48x88x88")
#     args = parser.parse_args()
#     os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
# else:
#     args = None
#
# import tensorflow as tf
# from csbdeep.io import save_tiff_imagej_compatible
# from csbdeep.models import CARE
# from setup_inference import setup_inference
# from tensorflow.python.client import device_lib
#
# axes = "ZYX"
#
#
# def get_available_gpus():
#     local_device_protos = device_lib.list_local_devices()
#     return [x.name for x in local_device_protos if x.device_type == "GPU"]
#
#
# def do_inference(file_path, setup, model_queue):
#     if (setup["result_path"] / file_path.name).exists():
#         return
#
#     x = imread(str(file_path))
#     if model_queue.empty():
#         print("model_queue!")
#
#     model = model_queue.get()
#
#     restored = model.predict(x, axes)
#     model_queue.put(model)
#     restored = setup["postprocess"](restored)
#     save_tiff_imagej_compatible(str(setup["result_path"] / file_path.name), restored, axes)
#
# if __name__ == "__main__":
#     setup = get_care_setup(subpath=args.subpath, model_name=args.model_name)
#     available_gpus = get_available_gpus()
#     print("available_gpus", available_gpus)
#
#     model_queue = Queue()
#
#
#     for i in range(len(available_gpus)):
#         with tf.device(f"/GPU:{i}"):
#             model_queue.put(CARE(config=None, name=setup["model_name"], basedir=setup["model_basedir"]))
#
#
#     with ThreadPoolExecutor(2 * len(available_gpus)) as executor:
#         futs = [
#             executor.submit(do_inference, file_path, setup, model_queue) for file_path in setup["data_path"].glob(f"test/lr/*.tif")
#         ]
#
#         for fut in tqdm(as_completed(futs)):
#             pass
