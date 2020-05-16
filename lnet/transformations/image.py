import logging
import typing
from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy
import skimage.transform
import torch
from scipy.ndimage import zoom

from lnet.transformations.affine_utils import get_crops, get_lf_crop
from lnet.transformations.base import Transform

logger = logging.getLogger(__name__)


class Crop(Transform):
    def __init__(
        self,
        crop: Optional[Tuple[Tuple[Union[int, float], Optional[Union[int, float]]], ...]] = None,
        crop_fn: Optional[
            Callable[[Tuple[int, ...]], Tuple[Tuple[Union[int, float], Optional[Union[int, float]]], ...]]
        ] = None,
        **super_kwargs,
    ):
        super().__init__(**super_kwargs)
        if crop is not None and crop_fn is not None:
            raise ValueError("exclusive arguments: `crop` and `crop_fn`")
        elif crop_fn is None:
            assert all(len(c) == 2 for c in crop)
            self.crop = crop
            self.crop_fn = None
        else:
            self.crop = None
            self.crop_fn = crop_fn

    def apply_to_tensor(
        self, tensor: Union[numpy.ndarray, torch.Tensor], *, name: str, idx: int, meta: List[dict]
    ) -> Union[numpy.ndarray, torch.Tensor]:
        crop = self.crop or self.crop_fn(tensor.shape[2:])
        assert len(tensor.shape) - 1 == len(crop), (tensor.shape, crop)
        int_crop = [[None if cc is None else int(cc) for cc in c] for c in crop]
        if any([crop[i][j] is not None and crop[i][j] != cc for i, c in enumerate(int_crop) for j, cc in enumerate(c)]):
            raise ValueError(f"Crop contains fractions: {crop}")

        out = tensor[(slice(None),) + tuple(slice(c[0], c[1]) for c in int_crop)]
        logger.debug("Crop tensor: %s %s by %s to %s", name, tensor.shape, crop, out.shape)
        return out


class CropByCropName(Transform):
    def __init__(
        self,
        apply_to: str,
        crops: typing.Dict[str, Sequence[Sequence[Optional[int]]]] = None,
        meta: Optional[dict] = None,
    ):
        assert isinstance(apply_to, str), "str to check if tensor is a slice (needed for get_crops)"
        if crops is None and apply_to == "lf":
            assert meta is not None
            crops = {
                crop_name: get_lf_crop(crop_name, shrink=meta["shrink"], nnum=meta["nnum"], scale=meta["scale"])
                for crop_name in meta["crop_names"]
            }

        assert crops is not None
        super().__init__(apply_to=apply_to)
        self.crops = {}
        for crop_name, crop in crops.items():
            if apply_to == "ls_trf":
                # add z axis
                crop = list(crop)
                crop.insert(1, [0, None])

            self.crops[crop_name] = Crop(apply_to=apply_to, crop=crop)

    def apply_to_tensor(
        self, tensor: typing.Any, *, name: str, idx: int, meta: typing.List[dict]
    ) -> typing.Union[numpy.ndarray, torch.Tensor]:
        crop_name = meta[0][name]["crop_name"]
        assert all([tmeta[name]["crop_name"] == crop_name for tmeta in meta]), meta
        return self.crops[crop_name].apply_to_tensor(tensor=tensor, name=name, idx=idx, meta=meta)


class CropLSforDynamicTraining(Transform):
    def __init__(
        self, apply_to: str, lf_crops: typing.Dict[str, Sequence[Sequence[Optional[int]]]] = None, meta: dict = None
    ):
        assert meta is not None
        assert isinstance(apply_to, str), "str to check if tensor is a slice (needed for get_crops)"
        if lf_crops is None:
            lf_crops = {crop_name: None for crop_name in meta["crop_names"]}

        super().__init__(apply_to=apply_to)
        self.crops = {}
        for crop_name, lf_crop in lf_crops.items():
            assert crop_name in meta["crop_names"], (crop_name, meta["crop_names"])
            _, _, ls_out = get_crops(crop_name, lf_crop=lf_crop, meta=meta, for_slice="slice" in apply_to)
            self.crops[crop_name] = Crop(apply_to=apply_to, crop=ls_out)

    def apply_to_tensor(
        self, tensor: typing.Any, *, name: str, idx: int, meta: typing.List[dict]
    ) -> typing.Union[numpy.ndarray, torch.Tensor]:
        crop_name = meta[0][name]["crop_name"]
        assert all([tmeta[name]["crop_name"] == crop_name for tmeta in meta]), meta
        return self.crops[crop_name].apply_to_tensor(tensor=tensor, name=name, idx=idx, meta=meta)


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

    def apply_to_tensor(
        self, tensor: typing.Any, *, name: str, idx: int, meta: typing.List[dict]
    ) -> typing.Union[numpy.ndarray, torch.Tensor]:
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
        self.axis = axis if axis < 0 else axis + 1  # add batch dim for positive axis

    def apply_to_tensor(
        self, tensor: Union[numpy.ndarray, torch.Tensor], *, name: str, idx: int, meta: List[dict]
    ) -> Union[numpy.ndarray, torch.Tensor]:
        if isinstance(tensor, numpy.ndarray):
            return numpy.flip(tensor, axis=self.axis)
        elif isinstance(tensor, torch.Tensor):
            return tensor.flip([self.axis])
        else:
            raise NotImplementedError


class RandomlyFlipAxis(Transform):
    meta_key_format = "flip_axis_{}"

    def __init__(self, axis: int, **super_kwargs):
        super().__init__(**super_kwargs)
        self.axis = axis

    def edit_meta_before(self, meta: List[dict]) -> List[dict]:
        key = self.meta_key_format.format(self.axis)
        for m in meta:
            assert key not in m, (key, m)
            m[key] = numpy.random.uniform() > 0.5

        return meta

    def apply_to_sample(
        self,
        sample: Union[numpy.ndarray, torch.Tensor],
        *,
        tensor_name: str,
        tensor_idx: int,
        batch_idx: int,
        meta: dict,
    ) -> Union[numpy.ndarray, torch.Tensor]:
        key = self.meta_key_format.format(self.axis)
        if meta[key]:
            if isinstance(sample, numpy.ndarray):
                return numpy.flip(sample, axis=self.axis)
            elif isinstance(sample, torch.Tensor):
                return sample.flip([self.axis])
            else:
                raise NotImplementedError

        return sample


class RandomIntensityScale(Transform):
    meta_key = "random_intensity_scaling"

    def __init__(self, factor_min: float = 0.9, factor_max: float = 1.1, **super_kwargs):
        super().__init__(**super_kwargs)
        self.factor_min = factor_min
        self.factor_max = factor_max

    def edit_meta_before(self, meta: List[dict]) -> List[dict]:
        for m in meta:
            assert self.meta_key not in m, m
            m[self.meta_key] = numpy.random.uniform(low=self.factor_min, high=self.factor_max)

        return meta

    def apply_to_sample(
        self,
        sample: Union[numpy.ndarray, torch.Tensor],
        *,
        tensor_name: str,
        tensor_idx: int,
        batch_idx: int,
        meta: dict,
    ):
        return sample * meta[self.meta_key]


class RandomRotate90(Transform):
    randomly_changes_shape = True
    meta_key = "random_rotate_90"

    def __init__(self, sample_axes: Tuple[int, int] = (-2, -1), **super_kwargs):
        super().__init__(**super_kwargs)
        self.sample_axes = sample_axes

    def edit_meta_before(self, meta: List[dict]) -> List[dict]:
        value = numpy.random.randint(4)  # same for whole batch
        for m in meta:
            assert self.meta_key not in m, m
            m[self.meta_key] = value

        return meta

    def apply_to_sample(
        self,
        sample: Union[numpy.ndarray, torch.Tensor],
        *,
        tensor_name: str,
        tensor_idx: int,
        batch_idx: int,
        meta: dict,
    ):
        return numpy.rot90(sample, k=meta[self.meta_key], axes=self.sample_axes)


class Zoom(Transform):
    def __init__(self, shape: Sequence[Union[int, float]], order: int, **super_kwargs):
        super().__init__(**super_kwargs)
        self.shape = shape
        assert 0 <= order <= 5, order
        self.order = order

    def apply_to_sample(
        self,
        sample: typing.Union[numpy.ndarray, torch.Tensor],
        *,
        tensor_name: str,
        tensor_idx: int,
        batch_idx: int,
        meta: dict,
    ) -> typing.Union[numpy.ndarray, torch.Tensor]:

        zoom_factors = [sout if isinstance(sout, float) else sout / sin for sin, sout in zip(sample.shape, self.shape)]
        out = zoom(sample, zoom_factors, order=self.order)
        logger.debug("Resize sample: %s %s by %s to %s", tensor_name, sample.shape, zoom_factors, out.shape)
        return out


class Resize(Transform):
    def __init__(self, shape: Sequence[Union[int, float]], order: int, **super_kwargs):
        super().__init__(**super_kwargs)
        self.shape = shape
        assert 0 <= order <= 5, order
        self.order = order

    def apply_to_sample(
        self,
        sample: Union[numpy.ndarray, torch.Tensor],
        *,
        tensor_name: str,
        tensor_idx: int,
        batch_idx: int,
        meta: dict,
    ):
        assert len(sample.shape) == len(self.shape), (sample.shape, self.shape)

        out_shape_float = [
            sout * sin if isinstance(sout, float) else sout for sin, sout in zip(sample.shape, self.shape)
        ]
        out_shape = [round(s) for s in out_shape_float]
        if out_shape_float != out_shape:
            logger.warning(
                "Resize sample %s (orig. size: %s) to rounded %s = %s",
                tensor_name,
                sample.shape,
                out_shape_float,
                out_shape,
            )

        logger.debug("Resize sample: %s %s by %s to %s", tensor_name, sample.shape, self.shape, out_shape)
        out = skimage.transform.resize(sample, out_shape, order=self.order, preserve_range=True)
        return out


# for debugging purposes:
class SetPixelValue(Transform):
    def __init__(self, value: float, **super_kwargs):
        super().__init__(**super_kwargs)
        self.value = value

    def apply_to_tensor(
        self, tensor: typing.Any, *, name: str, idx: int, meta: typing.List[dict]
    ) -> typing.Union[numpy.ndarray, torch.Tensor]:
        tensor[...] = self.value
        return tensor
