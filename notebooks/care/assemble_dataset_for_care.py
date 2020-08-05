import logging.config
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Sequence

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
            assert tensor.shape == shape
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
            assert not file_path.exists(), file_path
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
    parser = ArgumentParser(description="care prepper")
    parser.add_argument("config_path", type=Path)
    parser.add_argument("names", nargs="+")
    parser.add_argument("--data_root", type=Path, default=Path("/scratch/beuttenm/lnet/care"))

    args = parser.parse_args()
    config_path = args.config_path.absolute()
    logger.info("config_path: %s", config_path)
    assert config_path.exists()
    data_path = args.data_root / config_path.stem
    config = yaml.load(args.config_path)
    logger.info("selection: %s", args.names)
    for name in args.names:
        sub_config = config
        sub_data_path = data_path
        for name_part in name.split("/"):
            sub_config = sub_config[name_part]
            sub_data_path /= name_part

        logger.info("save to: %s", sub_data_path)
        try:
            assemble_dataset_for_care(
                sub_config, data_path=sub_data_path, names=("lr", "ls_reg" if config_path.stem == "beads" else "ls_trf")
            )
        except Exception as e:
            logger.error(e, exc_info=True)
