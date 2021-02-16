import logging
from typing import Any, Callable, Collection, Dict, Optional, Sequence, Tuple, Union

import numpy
import skimage.transform
import skimage.transform
import torch

from hylfm.utils.for_log import DuplicateLogFilter
from .affine_utils import get_lf_roi_in_raw_lf, get_ls_roi
from .base import Transform
from ..hylfm_types import Array

logger = logging.getLogger(__name__)


class Crop(Transform):
    def __init__(
        self,
        *,
        crop: Optional[Tuple[Tuple[Optional[int], Optional[int]], ...]] = None,
        crop_fn: Optional[Callable[[Tuple[int, ...]], Tuple[Tuple[int, int], ...]]] = None,
        apply_to: Union[str, Dict[str, str]],
    ):
        super().__init__(apply_to=apply_to)
        if (crop is not None and crop_fn is not None) or (crop is None and crop_fn is None):
            raise ValueError("exclusive arguments: `crop` and `crop_fn`")
        elif crop_fn is None:
            # assert all(len(c) == 2 for c in crop)
            self.crop_fn = None
            self.crop = crop
        else:
            self.crop_fn = crop_fn
            self.crop = None

    def apply_to_sample(self, tensor: Sequence) -> Union[numpy.ndarray, torch.Tensor]:
        if not isinstance(tensor, (numpy.ndarray, torch.Tensor)):
            raise TypeError(type(tensor))

        crop = self.crop if self.crop_fn is None else self.crop_fn(tensor.shape)
        assert len(tensor.shape) == len(crop), (tensor.shape, crop)
        return tensor[tuple(slice(lower, upper) for lower, upper in crop)]


class RandomlyFlipAxis(Transform):
    randomly_changes_shape = True

    def __init__(self, axis: int, **super_kwargs):
        super().__init__(**super_kwargs)
        self.axis = axis

    def apply_to_sample(self, **sample_tensors: Union[numpy.ndarray, torch.Tensor]) -> Dict[str, Any]:
        if numpy.random.uniform() < 0.5:
            for key in sample_tensors:
                if isinstance(sample_tensors[key], numpy.ndarray):
                    sample_tensors[key] = numpy.flip(sample_tensors[key], axis=self.axis)
                elif isinstance(sample_tensors[key], torch.Tensor):
                    sample_tensors[key] = sample_tensors[key].flip([self.axis])
                else:
                    raise NotImplementedError

        return sample_tensors


class RandomIntensityScale(Transform):
    def __init__(self, factor_min: float, factor_max: float, independent: bool, **super_kwargs):
        super().__init__(**super_kwargs)
        self.factor_min = factor_min
        self.factor_max = factor_max
        self.independent = independent

    def _get_factor(self):
        return numpy.random.uniform(low=self.factor_min, high=self.factor_max)

    def apply_to_sample(self, **sample_tensors: Array) -> Dict[str, Array]:
        factor = self._get_factor()
        for key, sample in sample_tensors.items():
            sample_tensors[key] = sample * factor
            if self.independent:
                factor = self._get_factor()

        return sample_tensors


class RandomRotate90(Transform):
    randomly_changes_shape = True

    def __init__(self, axes: Tuple[int, int] = (-2, -1), **super_kwargs):
        super().__init__(**super_kwargs)
        self.axes = [sa if sa < 0 else sa + 1 for sa in axes]  # add batch dim to axes

    def apply_to_batch(self, **batch: Array) -> Dict[str, Sequence]:
        k = numpy.random.randint(4)
        for key, tensor in batch.items():
            if isinstance(tensor, numpy.ndarray):
                batch[key] = numpy.rot90(tensor, k=k, axes=self.axes)
            else:
                raise NotImplementedError(type(tensor))

        return batch


class Resize(Transform):
    def __init__(self, shape: Sequence[Union[int, float]], order: int, apply_to: str):
        assert isinstance(apply_to, str)
        super().__init__(apply_to=apply_to)
        self.shape = shape
        assert 0 <= order <= 5, order
        self.order = order

        self.log_filter = DuplicateLogFilter()

    def apply_to_sample(self, tensor):
        assert len(tensor.shape) == len(self.shape), (tensor.shape, self.shape)

        out_shape_float = [
            sin if sout is None else sout * sin if isinstance(sout, float) else sout
            for sin, sout in zip(tensor.shape, self.shape)
        ]
        out_shape = [round(s) for s in out_shape_float]
        if out_shape_float != out_shape:
            logger = logging.Logger(self.__class__.__name__)
            logger.addFilter(self.log_filter)
            logger.warning(
                "Resize tensor (orig. size: %s) to rounded %s = %s", tensor.shape, out_shape_float, out_shape
            )

        # logger.debug("Resize tensor: %s by %s to %s", tensor.shape, self.shape, out_shape)

        out = skimage.transform.resize(tensor, out_shape, order=self.order, preserve_range=True)
        return out


class SelectRoi(Transform):
    def __init__(self, roi: Sequence[Union[int, None, slice]], apply_to: str):
        assert isinstance(apply_to, str)
        super().__init__(apply_to=apply_to)
        self.roi = tuple(self._slice_descr_to_slice(r) for r in roi)

    @staticmethod
    def _slice_descr_to_slice(slice_descr: Union[int, None, str]):
        if isinstance(slice_descr, slice):
            return slice_descr
        elif slice_descr is None:
            return slice(None)
        elif isinstance(slice_descr, int):
            return slice_descr
        else:
            raise NotImplementedError(slice_descr)

    def apply_to_sample(self, tensor):
        return tensor[self.roi]


class Transpose(Transform):
    def __init__(self, axes: Sequence[int], apply_to: str):
        assert isinstance(apply_to, str)
        super().__init__(apply_to=apply_to)
        self.axes = axes

    def apply_to_sample(self, tensor):
        if isinstance(tensor, numpy.ndarray):
            return tensor.transpose(self.axes)
        else:
            raise NotImplementedError(type(tensor))


class CropLSforDynamicTraining(Transform):
    def __init__(self, apply_to: str, crop_names: Collection[str], nnum: int, scale: int, z_ls_rescaled: int):
        assert isinstance(apply_to, str)
        super().__init__(
            input_mapping={apply_to: "tensor", "crop_name": "crop_name"}, output_mapping={"tensor": apply_to}
        )
        self.crops = {}
        for crop_name in crop_names:
            ls_roi = get_ls_roi(
                crop_name,
                nnum=nnum,
                for_slice="slice" in apply_to,
                wrt_ref=False,
                z_ls_rescaled=z_ls_rescaled,
                ls_scale=scale,
            )
            ls_roi = ((0, None),) + ls_roi  # add channel dim
            self.crops[crop_name] = Crop(apply_to=apply_to, crop=ls_roi)

    def apply_to_sample(self, tensor: Any, crop_name: str) -> Union[numpy.ndarray, torch.Tensor]:
        return self.crops[crop_name].apply_to_sample(tensor=tensor)


class CropWhatShrinkDoesNot(Transform):
    def __init__(self, apply_to: str, crop_names: Collection[str], nnum: int, scale: int, shrink: int, wrt_ref: bool):
        assert isinstance(apply_to, str)

        super().__init__(
            input_mapping={apply_to: "tensor", "crop_name": "crop_name"}, output_mapping={"tensor": apply_to}
        )
        self.crops = {}
        for crop_name in crop_names:
            roi = get_lf_roi_in_raw_lf(crop_name, nnum=nnum, shrink=shrink, scale=scale, wrt_ref=wrt_ref)
            if apply_to != "lf":
                roi = ((0, None),) + roi  # add z dim

            roi = ((0, None),) + roi  # add channel dim
            self.crops[crop_name] = Crop(apply_to=apply_to, crop=roi)

    def apply_to_sample(self, tensor: Array, crop_name: str) -> Union[numpy.ndarray, torch.Tensor]:
        return self.crops[crop_name].apply_to_sample(tensor=tensor)


class Pad(Transform):
    def __init__(self, pad_width: Sequence[Sequence[int]], pad_mode: str, nnum: Optional[int] = None, **super_kwargs):
        super().__init__(**super_kwargs)
        if any([len(p) != 2 for p in pad_width]) or any([pw < 0 for p in pad_width for pw in p]):
            raise ValueError(f"invalid pad_width sequence: {pad_width}")

        if pad_mode == "lenslets":
            if nnum is None:
                raise ValueError("nnum required to pad lenslets")
        else:
            raise NotImplementedError(pad_mode)

        self.pad_width = pad_width
        self.pad_mode = pad_mode
        self.nnum = nnum

    def apply_to_sample(self, tensor: Any) -> Union[numpy.ndarray, torch.Tensor]:
        assert len(tensor.shape) - 1 == len(self.pad_width)
        if isinstance(tensor, numpy.ndarray):
            if self.pad_mode == "lenslets":
                for i, (pw0, pw1) in enumerate(self.pad_width):
                    if pw0:
                        border_lenslets = tensor[(slice(None),) * (i + 1) + (slice(0, pw0 * self.nnum),)]
                        tensor = numpy.concatenate([border_lenslets, tensor], axis=i + 1)
                    if pw1:
                        border_lenslets = tensor[(slice(None),) * (i + 1) + (slice(-pw1 * self.nnum, None),)]
                        tensor = numpy.concatenate([tensor, border_lenslets], axis=i + 1)

                return tensor
            else:
                raise NotImplementedError(self.pad_mode)
                # return numpy.pad(tensor, pad_width=)
        else:
            NotImplementedError(type(tensor))


class FlipAxis(Transform):
    def __init__(self, axis: int, **super_kwargs):
        super().__init__(**super_kwargs)
        assert axis != 0, "You are not supposed to flip the batch dimension!"
        self.axis = axis

    def apply_to_batch(self, tensor: Union[numpy.ndarray, torch.Tensor]) -> Union[numpy.ndarray, torch.Tensor]:
        if isinstance(tensor, numpy.ndarray):
            return numpy.flip(tensor, axis=self.axis)
        elif isinstance(tensor, torch.Tensor):
            return tensor.flip([self.axis])
        else:
            raise NotImplementedError


# for debugging purposes:
class SetPixelValue(Transform):
    def __init__(self, value: float, **super_kwargs):
        super().__init__(**super_kwargs)
        self.value = value

    def apply_to_sample(self, tensor: Any) -> Union[numpy.ndarray, torch.Tensor]:
        tensor[...] = self.value
        return tensor
