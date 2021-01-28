from importlib import import_module
from typing import Union

from .base import (
    ConcatDataset,
    N5CachedDatasetFromInfo,
    N5CachedDatasetFromInfoSubset,
    TensorInfo,
    ZipDataset,
    get_dataset_from_info,
)
from .collate import collate, get_collate, separate
from .online import OnlineTensorInfo


def get_tensor_info(info_name: str, name: str, meta: dict) -> Union[TensorInfo, OnlineTensorInfo]:
    if info_name.startswith("local."):
        module_name_end = len("local.")
    else:
        module_name_end = 0

    module_name_end += info_name[module_name_end:].find(".")

    info_module_name = info_name[:module_name_end]
    tag = info_name[1 + module_name_end :]
    info_module = import_module("." + info_module_name, "hylfm.datasets")
    get_info = getattr(info_module, "get_tensor_info")
    return get_info(tag=tag, name=name, meta=meta)
