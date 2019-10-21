from lnet.utils.transforms import (
    Lightfield2Channel,
    RandomFlipXYnotZ,
    RandomRotate,
)

from inferno.io.transform.generic import Cast

from lnet.normalizations import norm00
from lnet.noises import noise00


def check_kwargs(ret, kwargs):
    if kwargs:
        raise ValueError(f"got unexpected kwargs: {kwargs}")

    return ret

known_transforms = {
    "norm00": lambda config, kwargs: check_kwargs(norm00, kwargs),
    "noise00": lambda config, kwargs: check_kwargs(noise00, kwargs),
    "RandomRotate": lambda config, kwargs: RandomRotate(**kwargs),
    "RandomFlipXYnotZ": lambda config, kwargs: RandomFlipXYnotZ(**kwargs),
    "Lightfield2Channel": lambda config, kwargs: Lightfield2Channel(nnum=config.model.nnum, **kwargs),
    "Cast": lambda config, kwargs: Cast(dtype=kwargs.pop("dtype", config.model.precision), **kwargs),
}
