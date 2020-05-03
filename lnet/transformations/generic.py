from typing import Any, Dict, List, Optional, Union

import numpy
import torch
import typing
from scipy.special import expit

from lnet.transformations.base import DTypeMapping, Transform


class AddConstant(Transform):
    def __init__(self, value: float, **super_kwargs):
        super().__init__(**super_kwargs)
        self.value = value

    def apply_to_tensor(self, tensor: numpy.ndarray, *, name: str, idx: int, meta: List[dict]):
        return tensor + self.value


class Cast(Transform, DTypeMapping):
    """Casts inputs to a specified datatype."""

    def __init__(
        self,
        dtype: str,
        device: str,
        numpy_kwargs: Optional[Dict[str, Any]] = None,
        non_blocking: bool = False,
        **super_kwargs,
    ):
        assert device in ("cpu", "cuda"), device
        if device == "cpu" and numpy_kwargs:
            raise ValueError(f"got numpy kwargs {numpy_kwargs}, but device != 'cpu'")

        super().__init__(**super_kwargs)
        self.dtype = self.DTYPE_MAPPING[dtype]
        self.device = device
        self.numpy_kwargs = numpy_kwargs or {}
        self.non_blocking = non_blocking

    def apply_to_tensor(self, tensor: numpy.ndarray, *, name: str, idx: int, meta: List[dict]):
        if isinstance(tensor, torch.Tensor):
            if self.device == "numpy":
                return tensor.detach().cpu().numpy().astype(self.dtype)
            else:
                return tensor.to(
                    dtype=getattr(torch, self.dtype), device=torch.device(self.device), non_blocking=self.non_blocking
                )
        elif isinstance(tensor, numpy.ndarray):
            if self.device == "numpy":
                assert not self.non_blocking, "'non_blocking' not supported for numpy.ndarray"
                return tensor.astype(self.dtype)
            elif self.device == "cuda":
                return torch.from_numpy(tensor.astype(dtype=self.dtype)).to(
                    dtype=getattr(torch, self.dtype), device=torch.device(self.device)
                )
            elif self.device == "cpu":
                assert not self.non_blocking, "'non_blocking' not supported for numpy.ndarray"
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


class Assert(Transform):
    def __init__(self, expected_tensor_shape: typing.Sequence[Optional[int]], **super_kwargs):
        super().__init__(**super_kwargs)
        self.expected_shape = expected_tensor_shape

    def apply_to_tensor(
        self, tensor: Any, *, name: str, idx: int, meta: typing.List[dict]
    ) -> Union[numpy.ndarray, torch.Tensor]:
        shape_is = tuple(tensor.shape)
        if len(shape_is) != len(self.expected_shape):
            raise ValueError(f"expected shape {self.expected_shape}, but found {shape_is}")

        for si, s in zip(shape_is, self.expected_shape):
            if s is None and si > 0:
                continue
            elif si != s:
                raise ValueError(f"expected shape {self.expected_shape}, but found {shape_is}")

        return tensor
