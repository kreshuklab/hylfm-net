import logging.config
import os
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Sequence

if __name__ == "__main__":
    os.nice(21)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy
import tifffile
from ruamel.yaml import YAML

from lnet.setup.base import DatasetGroupSetup

logging.config.dictConfig(
    {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "%(asctime)s %(name)s %(levelname)s %(message)s",  # .%(msecs)03d [%(processName)s/%(threadName)s]
                "datefmt": "%H:%M:%S",
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
        "loggers": {
            "": {"handlers": ["default"], "level": "INFO", "propagate": True},
            "lensletnet.datasets": {"handlers": ["default"], "level": "INFO", "propagate": False},
            "lensletnet": {"handlers": ["default"], "level": "INFO", "propagate": False},
        },
    }
)
logger = logging.getLogger(__name__)

yaml = YAML(typ="safe")


def volwrite(p: Path, data, compress=2, **kwargs):
    with p.open("wb") as f:
        tifffile.imsave(f, data, compress=compress, **kwargs)


def assemble_dataset_for_care(config: Dict[str, Any], data_path: Path, names: Sequence[str] = ("lr", "ls_trf")):
    data_path.mkdir(parents=True, exist_ok=True)
    assert names
    for name in names:
        (data_path / name).mkdir(exist_ok=True)

    group_setup = DatasetGroupSetup(batch_size=1, **config)
    dataset = group_setup.dataset
    #     data_loader = torch.utils.data.DataLoader(
    #                 dataset=group_setup.dataset,
    #                 shuffle=False,
    #                 collate_fn=get_collate_fn(lambda batch: batch),
    #                 num_workers=16,
    #                 pin_memory=False,
    #             )

    def save_image_pair(idx):
        tensors = dataset[idx]
        shape = tensors[names[0]].shape
        for name in names:
            tensor = tensors[name]
            assert isinstance(tensor, numpy.ndarray)
            assert tensor.shape == shape, {name: tensors[name].shape for name in names}
            if tensor.dtype == numpy.float64:
                tensor = tensor.astype(numpy.float32)
            else:
                assert tensor.dtype == numpy.float32
                # assert tensor.dtype == numpy.uint16, (tensor.dtype, tensor.min, tensor.max)

            assert len(tensor.shape) == 5
            assert tensor.shape[0] == 1  # batch dim
            assert tensor.shape[1] == 1  # channel dim
            tensor = tensor.squeeze()
            assert len(tensor.shape) == 3

            file_path = data_path / name / f"{idx:05}.tif"
            # assert not file_path.exists(), file_path
            if not file_path.exists():
                volwrite(file_path, tensor)

    with ThreadPoolExecutor(max_workers=16) as executor:
        futs = []
        for idx in range(len(dataset)):
            futs.append(executor.submit(save_image_pair, idx))

        for fut in as_completed(futs):
            exc = fut.exception()
            if exc is not None:
                raise exc


if __name__ == "__main__":
    parser = ArgumentParser(description="assemble dataset")
    parser.add_argument("config_path", type=Path)
    parser.add_argument("subpaths", nargs="+")
    parser.add_argument("--test_data_root", type=Path, default=Path("/g/kreshuk/LF_computed/lnet/plain"))
    parser.add_argument("--train_data_root", type=Path, default=Path("/scratch/beuttenm/lnet/plain"))

    args = parser.parse_args()
    config_path = args.config_path.absolute()

    test_data_root = args.test_data_root / config_path.stem
    train_data_root = args.train_data_root / config_path.stem

    logger.info("config_path: %s", config_path)
    assert config_path.exists()
    config = yaml.load(args.config_path)
    for subpath in args.subpaths:
        logger.info("subpath %s", subpath)
        if subpath.split("/")[-1] == "train":
            data_path = train_data_root / subpath
        elif subpath.split("/")[-1] == "test":
            data_path = test_data_root / subpath
        else:
            raise ValueError(subpath)

        logger.info("save %s to: %s", subpath, data_path)

        sub_config = config
        for name_part in subpath.split("/"):
            sub_config = sub_config[name_part]

        data_names = ["lr"]
        if "01highc" in subpath:
            data_names.append("ls_reg")
            data_names.append("hylfm")
        elif "dynamic" in subpath:
            data_names.append("ls_slice")
            data_names.append("hylfm_stat")
            data_names.append("hylfm_dyn")
        elif "static1" in subpath:
            data_names.append("ls_trf")
            data_names.append("hylfm_stat")
            data_names.append("hylfm_dyn")
        elif "static2" in subpath:
            data_names.append("ls_trf")
            data_names.append("hylfm_stat")
            data_names.append("hylfm_dyn")
        else:
            raise NotImplementedError(subpath)

        try:
            assemble_dataset_for_care(sub_config, data_path=data_path, names=tuple(data_names))
        except Exception as e:
            logger.error(e, exc_info=True)
            raise e
