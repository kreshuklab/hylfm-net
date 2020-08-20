import logging
import os
from argparse import ArgumentParser
from pathlib import Path

from ruamel.yaml import YAML
from tifffile import imread

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
