from copy import deepcopy
from importlib import import_module

from .base import (
    ConcatDataset,
    N5CachedDatasetFromInfo,
    N5CachedDatasetFromInfoSubset,
    get_collate_fn,
    ZipDataset,
    get_dataset_from_info,
    TensorInfo,
)


def get_tensor_info(info_name: str, name: str, meta: dict) -> TensorInfo:
    first_dot = info_name.find(".")
    info_module_name = info_name[:first_dot]
    tag = info_name[1 + first_dot :]
    info_module = import_module("." + info_module_name, "hylfm.datasets")
    info = deepcopy(getattr(info_module, tag, None))
    if info is None:
        info = getattr(info_module, "get_tensor_info")(tag=tag, name=name, meta=meta)

    return info
