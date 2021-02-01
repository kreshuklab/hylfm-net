import logging
from typing import Any, Dict, Optional, Sequence, Union

import numpy
import scipy.special
import torch

from .base import DTypeMapping, Transform
from ..hylfm_types import Array

logger = logging.getLogger(__name__)


class Identity(Transform):
    def __call__(self, tensors: Dict[str, Any]) -> Dict[str, Any]:
        return tensors


class AddConstant(Transform):
    def __init__(self, value: float, apply_to: Union[str, Dict[str, str]]):
        super().__init__(apply_to=apply_to)
        self.value = value

    def apply_to_batch(self, tensor) -> Dict[str, Any]:
        return tensor + self.value


class Cast(Transform, DTypeMapping):
    """Casts inputs to a specified datatype."""

    def __init__(
        self,
        *,
        dtype: str,
        device: str,
        numpy_kwargs: Optional[Dict[str, Any]] = None,
        non_blocking: bool = False,
        **super_kwargs,
    ):
        assert device == "numpy" or torch.device(device), device
        if device == "cpu" and numpy_kwargs:
            raise ValueError(f"got numpy kwargs {numpy_kwargs}, but device != 'cpu'")

        super().__init__(**super_kwargs)
        self.dtype = self.DTYPE_MAPPING[dtype]
        self.device = device
        self.numpy_kwargs = numpy_kwargs or {}
        self.non_blocking = non_blocking

    def apply_to_batch(self, **batch: Array) -> Dict[str, Array]:
        for key, tensor in batch.items():
            if isinstance(tensor, torch.Tensor):
                if self.device == "numpy":
                    batch[key] = tensor.detach().cpu().numpy().astype(self.dtype)
                else:
                    batch[key] = tensor.to(
                        dtype=getattr(torch, self.dtype),
                        device=torch.device(self.device),
                        non_blocking=self.non_blocking,
                    )
            elif isinstance(tensor, numpy.ndarray):
                if self.device == "numpy":
                    assert not self.non_blocking, "'non_blocking' not supported for numpy.ndarray"
                    batch[key] = tensor.astype(self.dtype, **self.numpy_kwargs)
                else:
                    batch[key] = torch.from_numpy(tensor.astype(dtype=self.dtype)).to(
                        dtype=getattr(torch, self.dtype), device=torch.device(self.device)
                    )
            else:
                raise NotImplementedError(type(tensor))

        return batch


class Clip(Transform):
    def __init__(
        self,
        *,
        min_: Optional[float] = None,
        max_: Optional[float] = None,
        as_local_percentile: bool = False,
        **super_kwargs,
    ):
        assert min_ is not None or max_ is not None
        super().__init__(**super_kwargs)
        self.min_ = min_
        self.max_ = max_
        self.as_local_percentile = as_local_percentile

    def apply_to_sample(self, tensor: Union[numpy.ndarray, torch.Tensor]):
        if isinstance(tensor, numpy.ndarray):
            if self.as_local_percentile:
                if self.min_ is None:
                    min_ = [None] * tensor.shape[0]
                else:
                    min_ = numpy.percentile(tensor, q=self.min_, axis=0)

                if self.max_ is None:
                    max_ = [None] * tensor.shape[0]
                else:
                    max_ = numpy.percentile(tensor, q=self.max_, axis=0)

                return numpy.stack([sample.clip(mi, ma) for sample, mi, ma in zip(tensor, min_, max_)])
            else:
                return tensor.clip(self.min_, self.max_)

        elif isinstance(tensor, torch.Tensor):
            if self.as_local_percentile:
                raise NotImplementedError("percentiles")
            else:
                return tensor.clamp(self.min_, self.max_)

        else:
            raise NotImplementedError(type(tensor))


class InsertSingletonDimension(Transform, DTypeMapping):
    def __init__(self, axis: int, **super_kwargs):
        super().__init__(**super_kwargs)
        self.axis = axis + 1

    def apply_to_sample(self, tensor: Union[numpy.ndarray, torch.Tensor]):
        if isinstance(tensor, torch.Tensor):
            return torch.unsqueeze(tensor, self.axis)
        elif isinstance(tensor, numpy.ndarray):
            return numpy.expand_dims(tensor, self.axis)
        else:
            raise NotImplementedError(type(tensor))


class RemoveSingletonDimension(Transform, DTypeMapping):
    def __init__(self, axis: int, **super_kwargs):
        super().__init__(**super_kwargs)
        self.axis = axis + 1

    def apply_to_sample(self, tensor: Union[numpy.ndarray, torch.Tensor]):
        try:
            if isinstance(tensor, torch.Tensor):
                return torch.squeeze(tensor, self.axis)
            elif isinstance(tensor, numpy.ndarray):
                return numpy.squeeze(tensor, self.axis)
            else:
                raise NotImplementedError(type(tensor))
        except Exception as e:
            logger.error(e)
            logger.error(f"%s %s %s", type(tensor), tensor.shape, self.axis)
            raise e


# class Squeeze(Transform, DTypeMapping):
#     def apply_to_sample(
#         self,
#         sample: Union[numpy.ndarray, torch.Tensor],
#         *,
#         tensor_name: str,
#         tensor_idx: int,
#         batch_idx: int,
#         meta: dict,
#     ) -> Union[numpy.ndarray, torch.Tensor]:
#         if isinstance(sample, torch.Tensor):
#             return torch.squeeze(sample)
#         elif isinstance(sample, numpy.ndarray):
#             return numpy.squeeze(sample)
#         else:
#             raise NotImplementedError(type(sample))


class Assert(Transform):
    def __init__(
        self,
        apply_to: str,
        *,
        expected_tensor_shape: Optional[Sequence[Optional[int]]] = None,
        expected_shape_like_tensor: Optional[str] = None,
    ):
        input_mapping = {apply_to: "tensor"}
        if (expected_tensor_shape is None and expected_shape_like_tensor is None) or (
            expected_tensor_shape is not None and expected_shape_like_tensor is not None
        ):
            raise ValueError("either expected_tensor_shape or expected_shape_like_tensor is required")

        if expected_tensor_shape is None:
            assert expected_shape_like_tensor not in input_mapping
            input_mapping[expected_shape_like_tensor] = "other"

        super().__init__(input_mapping=input_mapping)
        self.expected_shape = expected_tensor_shape
        self.expected_shape_like_tensor = expected_shape_like_tensor

    def apply_to_batch(self, tensor: Array, other: Optional[Array] = None) -> Dict[str, Any]:
        expected = self.expected_shape if self.expected_shape is not None else other.shape
        actual = tensor.shape
        if any(e != a for e, a in zip(expected, actual) if e is not None):
            raise AssertionError(f"shape mismatch: {actual} != {expected}")

        return {}


# class ToSimpleType(Transform):
#     def apply_to_sample(self, tensor: Any):
#         if isinstance(tensor, torch.Tensor):
#             if tensor.shape:
#                 return [self.apply_to_sample(t) for t in tensor]
#             else:
#                 return tensor.item()
#
#         elif isinstance(tensor, list):
#             return [self.apply_to_sample(t) for t in tensor]
#
#         raise NotImplementedError(tensor)


class Sigmoid(Transform):
    def apply_to_batch(self, tensor: Sequence) -> Union[numpy.ndarray, torch.Tensor]:
        if isinstance(tensor, numpy.ndarray):
            return scipy.special.expit(tensor)
        elif isinstance(tensor, torch.Tensor):
            return tensor.sigmoid()
        else:
            raise NotImplementedError(type(tensor))


class Softmax(Transform):
    def __init__(self, dim, apply_to: str):
        self.dim = dim if dim < 0 else dim + 1
        super().__init__(apply_to=apply_to)

    def apply_to_batch(self, tensor: Sequence) -> Union[numpy.ndarray, torch.Tensor]:
        if isinstance(tensor, numpy.ndarray):
            return scipy.special.softmax(tensor, axis=self.dim)
        elif isinstance(tensor, torch.Tensor):
            return torch.softmax(tensor, dim=self.dim)
        else:
            raise NotImplementedError(type(tensor))


class Argmax(Transform):
    def __init__(self, dim, apply_to: str):
        self.dim = dim if dim < 0 else dim + 1
        super().__init__(apply_to=apply_to)

    def apply_to_batch(self, tensor: Sequence) -> Union[numpy.ndarray, torch.Tensor]:
        if isinstance(tensor, numpy.ndarray):
            return numpy.argmax(tensor, axis=self.dim)
        elif isinstance(tensor, torch.Tensor):
            return torch.argmax(tensor, dim=self.dim)
        else:
            raise NotImplementedError(type(tensor))
