import logging
import os
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from ruamel.yaml import YAML
from tifffile import imread
from tqdm import tqdm

from plain.plain_setup import get_setup

logger = logging.getLogger(__name__)

yaml = YAML(typ="safe")


if __name__ == "__main__":
    parser = ArgumentParser(description="care inference")
    parser.add_argument("cuda_visible_device")
    parser.add_argument("subpath", default="heart/dynamic/v0_spe1000_on_48x88x88")
    parser.add_argument("file_paths_path", type=Path)
    args = parser.parse_args()
    assert "," not in args.cuda_visible_device
    assert args.cuda_visible_device

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_device
    from csbdeep.io import save_tiff_imagej_compatible
    from csbdeep.models import CARE

    # from tensorflow.python.client import device_lib

    axes = "ZYX"
    setup = get_setup(subpath=args.subpath)
    model = CARE(config=None, name=setup["model_path"].name, basedir=setup["model_path"].parent)

    for file_path in map(Path, yaml.load(args.file_paths_path)):
        result_path = file_path.parent.parent / setup["model_name"] / file_path.name
        if result_path.exists():
            continue

        x = imread(str(file_path))
        restored = model.predict(x, axes)
        restored = setup["postprocess"](restored)
        save_tiff_imagej_compatible(str(result_path), restored, axes)

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
# from get_care_setup import get_care_setup
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
