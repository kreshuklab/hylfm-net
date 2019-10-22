import numpy

from scipy.special import expit
from typing import List, Tuple

from inferno.io.transform import Transform, Compose
from inferno.io.transform.image import AdditiveGaussianNoise, RandomRotate, RandomTranspose


def lightfield_from_channel(cxy, nnum=19):
    c, x, y = cxy.shape
    assert c == nnum ** 2
    return cxy.reshape(nnum, nnum, x, y).transpose(2, 0, 3, 1).reshape(1, nnum * x, nnum * y)


def channel_from_lightfield(cxy, nnum=19):
    c, x, y = cxy.shape
    assert c == 1
    assert x % nnum == 0, (x, nnum)
    assert y % nnum == 0, (y, nnum)
    return cxy.reshape(x // nnum, nnum, y // nnum, nnum).transpose(1, 3, 0, 2).reshape(nnum ** 2, x // nnum, y // nnum)


class Clip(Transform):
    def __init__(self, min_: float, max_: float, apply_to: List[int], **super_kwargs):
        super().__init__(apply_to=apply_to, **super_kwargs)
        self.min_ = min_
        self.max_ = max_

    def tensor_function(self, tensor: numpy.ndarray):
        return tensor.clip(self.min_, self.max_)


class Normalize01(Transform):
    """Normalizes input by a constant."""

    def __init__(self, min_: float, max_: float, clip=True, **super_kwargs):
        """
        Parameters
        ----------
        min_ : min of input data
        max_ : max of input data
        super_kwargs : dict
            Kwargs to the superclass `inferno.io.transform.base.Transform`.
        """
        super().__init__(**super_kwargs)
        self.min_ = float(min_)
        self.range_ = float(max_) - self.min_
        self.clip = clip

    def tensor_function(self, tensor):
        ret = (tensor - self.min_) / (self.range_)
        if self.clip:
            ret = numpy.clip(ret, 0.0, 1.0)

        return ret


class NormalizeSigmoid(Transform):
    def __init__(self, mean: float, std: float, eps: float = 1e-4, **super_kwargs):
        super().__init__(**super_kwargs)
        self.mean = numpy.asarray(mean)
        self.std = numpy.asarray(std)
        self.eps = eps

    def tensor_function(self, tensor):
        # Figure out how to reshape mean and std
        reshape_as = [-1] + [1] * (tensor.ndim - 1)
        # Normalize
        tensor = (tensor - self.mean.reshape(*reshape_as)) / (self.std.reshape(*reshape_as) + self.eps)
        # apply sigmoid
        return expit(tensor)


class Normalize01Sig(Transform):
    """Normalizes input by a constant."""

    def __init__(self, min_: float, max_: float, **super_kwargs):
        """
        Parameters
        ----------
        min_ : min of input data
        max_ : max of input data
        super_kwargs : dict
            Kwargs to the superclass `inferno.io.transform.base.Transform`.
        """
        super().__init__(**super_kwargs)
        self.min_ = float(min_)
        self.max_ = float(max_)

    def tensor_function(self, tensor):
        return expit((4 * (tensor - self.min_) / (self.max_ - self.min_)) - 2)


class EdgeCrop(Transform):
    """Crop evenly from both edges of the last m axes for nD tensors with n >= m."""

    def __init__(self, crop: Tuple[int, ...], apply_to: List[int], **super_kwargs):
        super().__init__(apply_to=apply_to, **super_kwargs)
        self.crop = crop

    def tensor_function(self, tensor):
        return tensor[tuple([...] + [slice(c, -c) for c in self.crop])]


class RandomFlipXYnotZ(Transform):
    """Random flips along the last two axes for nD tensors with n >=2."""

    def __init__(self, flip_last_axis=True, flip_second_last_axis=True, **super_kwargs):
        super().__init__(**super_kwargs)
        self.flip_last = flip_last_axis
        self.flip_second_last = flip_second_last_axis

    def build_random_variables(self, **kwargs):
        numpy.random.seed()
        self.set_random_variable("last", numpy.random.uniform() > 0.5)
        self.set_random_variable("second_last", numpy.random.uniform() > 0.5)

    def tensor_function(self, image):
        if self.flip_last and self.get_random_variable("last"):
            image = image[..., ::-1]
        if self.flip_second_last and self.get_random_variable("second_last"):
            image = image[..., ::-1, :]
        return image


class Lightfield2Channel(Transform):
    def __init__(self, nnum: int, apply_to=(0,), **super_kwargs):
        super().__init__(apply_to=list(apply_to), **super_kwargs)
        self.nnum = nnum

    def tensor_function(self, cxy_image):
        return channel_from_lightfield(cxy_image, nnum=self.nnum)


def get_prereorder_transform(*additional_transforms, std: float = 0.1):
    transforms = [AdditiveGaussianNoise(std, apply_to=[0]), RandomRotate(), RandomFlipXYnotZ()]
    for t in additional_transforms:
        transforms.append(t)

    return Compose(*transforms)
