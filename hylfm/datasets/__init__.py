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
    first_dot = info_name.find(".")
    info_module_name = info_name[:first_dot]
    tag = info_name[1 + first_dot :]
    info_module = import_module("." + info_module_name, "hylfm.datasets")
    info = deepcopy(getattr(info_module, tag, None))
    if info is None:
        info = getattr(info_module, "get_tensor_info")(tag=tag, name=name, meta=meta)

    return info
