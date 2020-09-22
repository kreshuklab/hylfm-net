from copy import deepcopy
from importlib import import_module
from typing import Union

from .base import (
    ConcatDataset,
    N5CachedDatasetFromInfo,
    N5CachedDatasetFromInfoSubset,
    TensorInfo,
    ZipDataset,
    get_collate_fn,
    get_dataset_from_info,
)
from .online import OnlineTensorInfo


def get_tensor_info(info_name: str, name: str, meta: dict) -> Union[TensorInfo, OnlineTensorInfo]:
    if info_name.startswith("local."):
        module_name_end = len("local.") + info_name[len("local."):].find(".")
    else:
        module_name_end = info_name.find(".")

    info_module_name = info_name[:module_name_end]
    tag = info_name[1 + module_name_end :]
    info_module = import_module("." + info_module_name, "hylfm.datasets")
    get_info = getattr(info_module, "get_tensor_info")
    info = get_info(tag=tag, name=name, meta=meta)
    return info
