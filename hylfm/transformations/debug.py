# import logging
# from typing import Any, Dict, List, Optional, Union
#
# import numpy
# import torch
# import typing
# from scipy.special import expit
#
# from hylfm.transformations.base import DTypeMapping, Transform
#
#
# logger = logging.getLogger(__name__)
#
#
# class SaveToDisk(Transform):
#     def __init__(self, value: float, **super_kwargs):
#         super().__init__(**super_kwargs)
#         self.value = value
#
#     def apply_to_tensor(self, tensor: numpy.ndarray, *, name: str, idx: int, meta: List[dict]):
#         return tensor + self.value
#
#
# class Cast(Transform, DTypeMapping):
#     """Casts inputs to a specified datatype."""
#
#     def __init__(
#         self,
#         dtype: str,
#         device: str,
#         numpy_kwargs: Optional[Dict[str, Any]] = None,
#         non_blocking: bool = False,
#         **super_kwargs,
#     ):
#         assert device in ("cpu", "cuda", "numpy"), device
#         if device == "cpu" and numpy_kwargs:
#             raise ValueError(f"got numpy kwargs {numpy_kwargs}, but device != 'cpu'")
#
#         super().__init__(**super_kwargs)
#         self.dtype = self.DTYPE_MAPPING[dtype]
#         self.device = device
#         self.numpy_kwargs = numpy_kwargs or {}
#         self.non_blocking = non_blocking
#
#     def apply_to_tensor(self, tensor: numpy.ndarray, *, name: str, idx: int, meta: List[dict]):
#         if isinstance(tensor, torch.Tensor):
#             if self.device == "numpy":
#                 return tensor.detach().cpu().numpy().astype(self.dtype)
#             else:
#                 return tensor.to(
#                     dtype=getattr(torch, self.dtype), device=torch.device(self.device), non_blocking=self.non_blocking
#                 )
#         elif isinstance(tensor, numpy.ndarray):
#             if self.device == "numpy":
#                 assert not self.non_blocking, "'non_blocking' not supported for numpy.ndarray"
#                 return tensor.astype(self.dtype, **self.numpy_kwargs)
#             else:
#                 return torch.from_numpy(tensor.astype(dtype=self.dtype)).to(
#                     dtype=getattr(torch, self.dtype), device=torch.device(self.device)
#                 )
#         else:
#             raise NotImplementedError(type(tensor))
#
#
# class InsertSingletonDimension(Transform, DTypeMapping):
#     def __init__(self, axis: int, **super_kwargs):
#         super().__init__(**super_kwargs)
#         self.axis = axis + 1
#
#     def apply_to_tensor(self, tensor: Union[numpy.ndarray, torch.Tensor], *, name: str, idx: int, meta: List[dict]):
#         if isinstance(tensor, torch.Tensor):
#             return torch.unsqueeze(tensor, self.axis)
#         elif isinstance(tensor, numpy.ndarray):
#             return numpy.expand_dims(tensor, self.axis)
#         else:
#             raise NotImplementedError(type(tensor))
#
#
# class RemoveSingletonDimension(Transform, DTypeMapping):
#     def __init__(self, axis: int, **super_kwargs):
#         super().__init__(**super_kwargs)
#         self.axis = axis + 1
#
#     def apply_to_tensor(self, tensor: Union[numpy.ndarray, torch.Tensor], *, name: str, idx: int, meta: List[dict]):
#         try:
#             if isinstance(tensor, torch.Tensor):
#                 return torch.squeeze(tensor, self.axis)
#             elif isinstance(tensor, numpy.ndarray):
#                 return numpy.squeeze(tensor, self.axis)
#             else:
#                 raise NotImplementedError(type(tensor))
#         except Exception as e:
#             logger.error(e)
#             logger.error(f"%s %s %s", type(tensor), tensor.shape, self.axis)
#             raise e
#
#
# class Squeeze(Transform, DTypeMapping):
#     def apply_to_sample(
#         self,
#         sample: typing.Union[numpy.ndarray, torch.Tensor],
#         *,
#         tensor_name: str,
#         tensor_idx: int,
#         batch_idx: int,
#         meta: dict,
#     ) -> typing.Union[numpy.ndarray, torch.Tensor]:
#         if isinstance(sample, torch.Tensor):
#             return torch.squeeze(sample)
#         elif isinstance(sample, numpy.ndarray):
#             return numpy.squeeze(sample)
#         else:
#             raise NotImplementedError(type(sample))
#
#
#
# class Sigmoid(Transform):
#     def apply_to_tensor(self, tensor: Union[numpy.ndarray, torch.Tensor], *, name: str, idx: int, meta: List[dict]):
#         if isinstance(tensor, numpy.ndarray):
#             return expit(tensor)
#         elif isinstance(tensor, torch.Tensor):
#             return tensor.sigmoid()
#         else:
#             raise NotImplementedError(type(tensor))
#
#
# class Assert(Transform):
#     def __init__(self, expected_tensor_shape: Union[str, typing.Sequence[Optional[int]]], **super_kwargs):
#         super().__init__(**super_kwargs)
#         self.expected_shape = expected_tensor_shape
#
#     def apply(self, tensors: typing.OrderedDict[str, typing.Any]) -> typing.OrderedDict[str, typing.Any]:
#         if isinstance(self.expected_shape, str):
#             for at in self.apply_to:
#                 tensors["meta"][0][at]["Assert_expected_tensor_shape"] = tensors[self.expected_shape].shape[1:]
#
#         return super().apply(tensors)
#
#     def apply_to_tensor(
#         self, tensor: Any, *, name: str, idx: int, meta: typing.List[dict]
#     ) -> Union[numpy.ndarray, torch.Tensor]:
#         b, *shape_is = tensor.shape
#
#         expected_shape = (
#             meta[0][name]["Assert_expected_tensor_shape"]
#             if isinstance(self.expected_shape, str)
#             else self.expected_shape
#         )
#         expected_shape_from = f" (from {self.expected_shape})" if isinstance(self.expected_shape, str) else ""
#         if len(shape_is) != len(expected_shape):
#             raise ValueError(
#                 f"expected shape b,{expected_shape}{expected_shape_from}, but found b={b},{shape_is} for tensor {name}"
#             )
#
#         for si, s in zip(shape_is, expected_shape):
#             if s is None and si > 0:
#                 continue
#             elif si != s:
#                 raise ValueError(
#                     f"expected shape b,{expected_shape}{expected_shape_from}, but found b={b},{shape_is} for tensor {name}"
#                 )
#
#         logger.debug(f"{name} has expected shape: b={b},{shape_is}{expected_shape_from}")
#         return tensor
