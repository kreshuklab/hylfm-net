from lnet.utils.transforms_impl import (
    Lightfield2Channel,
    RandomFlipXYnotZ,
    RandomRotate,
)

from inferno.io.transform.generic import Cast

from lnet.normalizations import norm00
from lnet.noises import noise00

known_transforms = {
    "norm00": lambda config: norm00,
    "noise00": lambda config: noise00,
    "RandomRotate": lambda config: RandomRotate(),
    "RandomFlipXYnotZ": lambda config: RandomFlipXYnotZ(),
    "Lightfield2Channel": lambda config: Lightfield2Channel(nnum=config.model.nnum),
    "Cast": lambda config: Cast(config.model.precision),
}
