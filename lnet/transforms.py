from lnet.utils.data_transform import Lightfield2Channel
from lnet.utils.normalizations import *
from lnet.utils.noises import *

RandomRotate = lambda config: RandomRotate()
RandomFlipXYnotZ = lambda config: RandomFlipXYnotZ(),
Lightfield2Channel19 = lambda config: Lightfield2Channel(nnum=config.model.nnum)
Cast = lambda config: Cast(torch_dtype_to_inferno[config.model.precision])