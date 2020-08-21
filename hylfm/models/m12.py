# if __name__ == "__main__":
#     import sys
#     sys.path.append("/g/kreshuk/beuttenm/repos/lnet/")

import logging
from functools import partial
from typing import Optional, Tuple

import torch.nn
import torch.nn as nn
from inferno.extensions.initializers import Constant, Initialization

from lnet.models.base import LnetModel
from lnet.models.layers.conv_layers import Conv2D, ResnetBlock, ValidConv3D
from lnet.models.layers.structural_layers import C2Z

logger = logging.getLogger(__name__)


class M12(LnetModel):
    def __init__(self, z_out: int, nnum: int, final_activation: Optional[str] = None):
        super().__init__()
        inplanes = nnum ** 2
        planes = 64
        z_valid_cut = 10
        z_out += z_valid_cut
        self.res2d_1 = ResnetBlock(in_n_filters=inplanes, n_filters=planes, valid=False)
        self.res2d_2 = ResnetBlock(in_n_filters=planes, n_filters=planes, valid=False)
        self.res2d_3 = ResnetBlock(in_n_filters=planes, n_filters=planes, valid=False)

        inplanes = planes
        planes = z_out
        init = partial(
            Initialization,
            weight_initializer=partial(nn.init.xavier_uniform_, gain=nn.init.calculate_gain("relu")),
            bias_initializer=Constant(0.0),
        )
        self.conv2 = Conv2D(inplanes, planes, (3, 3), activation="ReLU", initialization=init)

        self.c2z = C2Z(z_out)
        inplanes = self.c2z.get_c_out(planes)
        planes = 64
        self.red3d_1 = ResnetBlock(in_n_filters=inplanes, n_filters=planes, kernel_size=(3, 3, 3), valid=True)
        self.transposed_conv_1 = nn.ConvTranspose3d(
            in_channels=planes,
            out_channels=planes,
            kernel_size=(3, 2, 2),
            stride=(1, 2, 2),
            padding=(1, 0, 0),
            output_padding=0,
        )
        self.red3d_2 = ResnetBlock(in_n_filters=planes, n_filters=planes, kernel_size=(3, 3, 3), valid=True)
        self.transposed_conv_2 = nn.ConvTranspose3d(
            in_channels=planes,
            out_channels=planes,
            kernel_size=(3, 2, 2),
            stride=(1, 2, 2),
            padding=(1, 0, 0),
            output_padding=0,
        )
        init = partial(
            Initialization,
            weight_initializer=partial(nn.init.xavier_uniform_, gain=nn.init.calculate_gain("linear")),
            bias_initializer=Constant(0.0),
        )
        self.out = ValidConv3D(planes, 1, (3, 3, 3), initialization=init)

        if final_activation == "sigmoid":
            self.final_activation = torch.nn.Sigmoid()
        elif final_activation is not None:
            raise NotImplementedError(final_activation)
        else:
            self.final_activation = None

    def forward(self, x):
        # logger.warning("m12 forward")
        # print(x.shape)
        x = self.res2d_1(x)
        # print(x.shape)
        x = self.res2d_2(x)
        # print(x.shape)
        x = self.res2d_3(x)
        # print(x.shape)
        x = self.conv2(x)
        # print(x.shape)
        x = self.c2z(x)
        # print(x.shape)
        x = self.red3d_1(x)
        # print(x.shape)
        x = self.transposed_conv_1(x)
        # print(x.shape)
        x = self.red3d_2(x)
        # print(x.shape)
        x = self.transposed_conv_2(x)
        # print(x.shape)
        # logger.warning("intermed done")
        out = self.out(x)
        # logger.warning("out done")

        if self.final_activation is not None:
            out = self.final_activation(out)

        # print('out', out.shape)
        return out

    @classmethod
    def get_scaling(cls, ipt_shape: Optional[Tuple[int, int]] = None) -> Tuple[float, float]:
        return 4.0, 4.0

    @classmethod
    def get_shrinkage(cls, ipt_shape: Optional[Tuple[int, int]] = None) -> Tuple[int, int]:
        return 13, 13

    def get_output_shape(self, ipt_shape: Tuple[int, int]):
        return tuple(
            [i * sc - 2 * sr for i, sc, sr in zip(ipt_shape, self.get_scaling(), self.get_shrinkage())]
        )

if __name__ == "__main__":
    # import sys
    # sys.path.append("/g/kreshuk/beuttenm/repos/lnet")
    ipt = torch.ones(1, 19**2, 10, 20)
    model = M12(z_out=7, nnum=19)
    print('srhink')
    print((ipt).shape)
