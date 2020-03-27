import typing
from collections import OrderedDict
from typing import Any, Dict, Optional, Sequence, Union, NewType

import numpy
import torch

try:
    from typing import Protocol
except ImportError:
    from typing_extensions import Protocol


class Transform:
    randomly_changes_shape: bool = False

    def __init__(self, apply_to: Optional[Union[str, int, Sequence[Union[str, int]]]] = None):
        if isinstance(apply_to, (int, str)):
            self.apply_to = [str(apply_to)]
        elif apply_to is None:
            self.apply_to = None
        else:
            self.apply_to = [str(at) for at in apply_to]

        # self._random_variables = {}

    # def build_random_variables(self, **kwargs):
    #     pass
    #
    # def clear_random_variables(self):
    #     self._random_variables = {}

    def __call__(self, tensors: typing.OrderedDict[str, Any]) -> typing.OrderedDict[str, Any]:
        if self.apply_to is not None:
            cant_apply = [at for at in self.apply_to if at not in tensors]
            if cant_apply:
                raise ValueError(f"`apply_to` keys {cant_apply} not found in tensors {tensors}")

        initial_meta = tensors.get("meta", None)
        meta = self.edit_meta_before(initial_meta)
        if meta is not None:
            tensors["meta"] = meta
        elif initial_meta is not None:
            tensors.pop("meta")

        transformed = self.apply(tensors)
        assert isinstance(transformed, OrderedDict)

        return transformed

    def edit_meta_before(self, meta: Optional[dict]) -> Optional[dict]:
        return meta

    def apply(self, tensors: typing.OrderedDict[str, Any]) -> typing.OrderedDict[str, Any]:
        apply_to = tensors.keys() if self.apply_to is None else self.apply_to
        return OrderedDict(
            [
                (n, self.apply_to_tensor(t, name=n, idx=i, meta=tensors.get("meta", None))) if n in apply_to else (n, t)
                for i, (n, t) in enumerate(tensors.items())
            ]
        )

    def apply_to_tensor(self, tensor: Any, *, name: str, idx: int, meta: Optional[dict]):

        if isinstance(tensor, numpy.ndarray):
            return numpy.stack(
                [
                    self.apply_to_sample(s, tensor_name=name, tensor_idx=idx, batch_idx=i, meta=meta)
                    for i, s in enumerate(tensor)
                ]
            )
        else:
            raise NotImplementedError

    def apply_to_sample(
        self,
        sample: Union[numpy.ndarray, torch.Tensor],
        *,
        tensor_name: str,
        tensor_idx: int,
        batch_idx: int,
        meta: Optional[dict],
    ):
        raise NotImplementedError

class TransformLike(Protocol):
    def __call__(self, tensors: typing.OrderedDict[str, Any]) -> typing.OrderedDict[str, Any]:
        pass


class ComposedTransform(Transform):
    def __init__(self, *transforms: TransformLike, **super_kwargs):
        assert all([callable(transform) for transform in transforms])
        super().__init__(**super_kwargs)
        self.transforms = list(transforms)

    def add(self, transform):
        assert callable(transform)
        self.transforms.append(transform)
        return self

    def remove(self, name):
        transform_idx = None
        for idx, transform in enumerate(self.transforms):
            if type(transform).__name__ == name:
                transform_idx = idx
                break
        if transform_idx is not None:
            self.transforms.pop(transform_idx)

        return self

    def apply(self, tensors: typing.OrderedDict[str, Any]):
        for transform in self.transforms:
            tensors = transform(tensors)
            assert isinstance(tensors, OrderedDict), transform

        return tensors


class DTypeMapping:
    DTYPE_MAPPING = {
        "float32": "float32",
        "float": "float32",
        "double": "float64",
        "float64": "float64",
        "half": "float16",
        "float16": "float16",
        "long": "int64",
        "int64": "int64",
        "byte": "uint8",
        "uint8": "uint8",
        "int": "int32",
        "int32": "int32",
    }


#
# known_transforms = {
#     "norm": lambda model_config, kwargs: norm(**kwargs),
#     "norm01": lambda model_config, kwargs: norm01(**kwargs),
#     "clip": lambda model_config, kwargs: Clip(**kwargs),
#     "additive_gaussian_noise": lambda model_config, kwargs: additive_gaussian_noise(**kwargs),
#     "RandomRotate": lambda model_config, kwargs: RandomRotate(**kwargs),
#     "RandomFlipXYnotZ": lambda model_config, kwargs: RandomFlipXYnotZ(**kwargs),
#     "Lightfield2Channel": lambda model_config, kwargs: Lightfield2Channel(nnum=model_config.nnum, **kwargs),
#     "Cast": lambda model_config, kwargs: Cast(dtype=kwargs.pop("dtype", model_config.precision), **kwargs),
#     "RandomIntensityScale": lambda model_config, kwargs: RandomIntensityScale(**kwargs),
# }
#
# randomly_shape_changing_transforms = {"RandomRotate", "RandomFlipXYnotZ"}
#
