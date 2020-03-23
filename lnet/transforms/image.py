from typing import Any, Callable, Optional, Tuple, Union

import numpy
import torch

from lnet.transforms.base import Transform


class EdgeCrop(Transform):
    """Crop evenly from both edges of the last m axes for nD tensors with n >= m."""

    def __init__(
        self,
        crop: Optional[Tuple[int, ...]] = None,
        crop_fn: Optional[Callable[[Tuple[int, ...]], Tuple[int, ...]]] = None,
        **super_kwargs
    ):
        super().__init__(**super_kwargs)
        if crop is not None and crop_fn is not None:
            raise ValueError("exclusive arguments: `crop` and `crop_fn`")
        elif crop_fn is None:
            self.crop_fn = lambda _: crop
        else:
            self.crop_fn = crop_fn

    def apply_to_tensor(self, tensor: Union[numpy.ndarray, torch.Tensor], *, name: str, idx: int, meta: Optional[dict]):
        crop = self.crop_fn(tensor.shape[2:])
        return tensor[(slice(None),) + tuple(slice(c, -c) for c in crop)]


class RandomlyFlipAxis(Transform):
    meta_key_format = "flip_axis_{}"

    def __init__(self, axis: int, **super_kwargs):
        super().__init__(**super_kwargs)
        self.axis = axis

    def edit_meta_before(self, meta: Optional[dict]) -> Optional[dict]:
        meta = meta or {}
        key = self.meta_key_format.format(self.axis)
        assert key not in meta
        meta[key] = numpy.random.uniform() > 0.5
        return meta

    def apply_to_tensor(self, tensor: Any, *, name: str, idx: int, meta: Optional[dict]):
        # additional check to avoid to randomly not raise
        if not isinstance(tensor, (numpy.ndarray, torch.Tensor)):
            raise NotImplementedError

        key = self.meta_key_format.format(self.axis)
        if meta[key]:
            if isinstance(tensor, numpy.ndarray):
                return numpy.flip(tensor, axis=self.axis)
            elif isinstance(tensor, torch.Tensor):
                return tensor.flip([self.axis])
            else:
                raise NotImplementedError


class RandomIntensityScale(Transform):
    meta_key = "random_intensity_scaling"

    def __init__(self, factor_min: float = 0.9, factor_max: float = 1.1, **super_kwargs):
        super().__init__(**super_kwargs)
        self.factor_min = factor_min
        self.factor_max = factor_max

    def edit_meta_before(self, meta: Optional[dict]) -> dict:
        meta = meta or {}
        assert self.meta_key not in meta, meta
        meta[self.meta_key] = numpy.random.uniform(low=self.factor_min, high=self.factor_max)
        return meta

    def apply_to_tensor(self, tensor: Any, *, name: str, idx: int, meta: Optional[dict]):
        return tensor * meta[self.meta_key]
