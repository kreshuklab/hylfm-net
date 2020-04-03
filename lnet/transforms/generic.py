from typing import Any, Dict, List, Optional, Union

import numpy
import torch
from scipy.special import expit

from lnet.transforms.base import DTypeMapping, Transform


class AddConstant(Transform):
    def __init__(self, value: float, **super_kwargs):
        super().__init__(**super_kwargs)
        self.value = value

    def apply_to_tensor(self, tensor: numpy.ndarray, *, name: str, idx: int, meta: List[dict]):
        return tensor + self.value


class Cast(Transform, DTypeMapping):
    """Casts inputs to a specified datatype."""

    def __init__(
        self, dtype="float32", device: str = "cpu", numpy_kwargs: Optional[Dict[str, Any]] = None, **super_kwargs
    ):
        super().__init__(**super_kwargs)
        self.dtype = self.DTYPE_MAPPING[dtype]
        assert device in ("cpu", "cuda"), device
        self.device = device
        self.numpy_kwargs = numpy_kwargs

    def apply_to_tensor(self, tensor: numpy.ndarray, *, name: str, idx: int, meta: List[dict]):
        if isinstance(tensor, torch.Tensor):
            return tensor.to(dtype=getattr(torch, self.dtype), device=torch.device(self.device))
        elif isinstance(tensor, numpy.ndarray):
            if self.device == "cuda":
                return torch.from_numpy(tensor).to(dtype=getattr(torch, self.dtype), device=torch.device(self.device))
            elif self.device == "cpu":
                return tensor.astype(self.dtype, **self.numpy_kwargs)
            else:
                raise NotImplementedError(self.device)
        else:
            raise NotImplementedError(type(tensor))


class Clip(Transform):
    def __init__(self, min_: float, max_: float, **super_kwargs):
        super().__init__(**super_kwargs)
        self.min_ = min_
        self.max_ = max_

    def apply_to_tensor(self, tensor: Union[numpy.ndarray, torch.Tensor], *, name: str, idx: int, meta: List[dict]):
        if isinstance(tensor, numpy.ndarray):
            return tensor.clip(self.min_, self.max_)
        elif isinstance(tensor, torch.Tensor):
            return tensor.clamp(self.min_, self.max_)
        else:
            raise NotImplementedError(type(tensor))


class Sigmoid(Transform):
    def apply_to_tensor(self, tensor: Union[numpy.ndarray, torch.Tensor], *, name: str, idx: int, meta: List[dict]):
        if isinstance(tensor, numpy.ndarray):
            return expit(tensor)
        elif isinstance(tensor, torch.Tensor):
            return tensor.sigmoid()
        else:
            raise NotImplementedError(type(tensor))
