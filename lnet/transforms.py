from lnet.utils.transforms import Lightfield2Channel, RandomFlipXYnotZ, RandomRotate

from inferno.io.transform.generic import Cast

from lnet.normalizations import norm, norm01
from lnet.noises import additive_gaussian_noise


known_transforms = {
    "norm": lambda config, kwargs: norm(**kwargs),
    "norm01": lambda config, kwargs: norm01(**kwargs),
    "additive_gaussian_noise": lambda config, kwargs: additive_gaussian_noise(**kwargs),
    "RandomRotate": lambda config, kwargs: RandomRotate(**kwargs),
    "RandomFlipXYnotZ": lambda config, kwargs: RandomFlipXYnotZ(**kwargs),
    "Lightfield2Channel": lambda config, kwargs: Lightfield2Channel(nnum=config.model.nnum, **kwargs),
    "Cast": lambda config, kwargs: Cast(dtype=kwargs.pop("dtype", config.model.precision), **kwargs),
}
