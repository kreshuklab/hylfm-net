from lnet.utils.transforms import Lightfield2Channel, RandomFlipXYnotZ, RandomRotate

from inferno.io.transform.generic import Cast

from .normalizations import norm, norm01
from .noises import additive_gaussian_noise


known_transforms = {
    "norm": lambda model_config, kwargs: norm(**kwargs),
    "norm01": lambda model_config, kwargs: norm01(**kwargs),
    "additive_gaussian_noise": lambda model_config, kwargs: additive_gaussian_noise(**kwargs),
    "RandomRotate": lambda model_config, kwargs: RandomRotate(**kwargs),
    "RandomFlipXYnotZ": lambda model_config, kwargs: RandomFlipXYnotZ(**kwargs),
    "Lightfield2Channel": lambda model_config, kwargs: Lightfield2Channel(nnum=model_config.nnum, **kwargs),
    "Cast": lambda model_config, kwargs: Cast(dtype=kwargs.pop("dtype", model_config.precision), **kwargs),
}

randomly_shape_changing_transforms = {"RandomRotate", "RandomFlipXYnotZ"}
