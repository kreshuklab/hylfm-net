import logging
from functools import partial
from typing import Optional, Tuple

import torch.nn
import torch.nn as nn
from inferno.extensions.initializers import Constant, Initialization

from hylfm.models.base import LnetModel
from hylfm.models.layers.conv_layers import Conv2D, ResnetBlock, ValidConv3D
from hylfm.models.layers.structural_layers import C2Z

logger = logging.getLogger(__name__)


class M16(LnetModel):
    def __init__(
        self, *, input_name: str, prediction_name: str, z_out: int, nnum: int, final_activation: Optional[str] = None
    ):
        super().__init__()
        self.input_name = input_name
        self.prediction_name = prediction_name
        inplanes = nnum ** 2
        z_valid_cut = 10
        z_out += z_valid_cut
        self.res2d_1 = ResnetBlock(in_n_filters=inplanes, n_filters=256, valid=False)
        self.res2d_2 = ResnetBlock(in_n_filters=256, n_filters=128, valid=False)
        self.res2d_3 = ResnetBlock(in_n_filters=128, n_filters=z_out * 2, valid=False)

        init = partial(
            Initialization,
            weight_initializer=partial(nn.init.xavier_uniform_, gain=nn.init.calculate_gain("relu")),
            bias_initializer=Constant(0.0),
        )
        self.conv2 = Conv2D(z_out * 2, z_out * 2, (3, 3), activation="ReLU", initialization=init)

        self.c2z = C2Z(z_out)
        inplanes = self.c2z.get_c_out(z_out * 2)
        self.red3d_1 = ResnetBlock(in_n_filters=inplanes, n_filters=32, kernel_size=(3, 3, 3), valid=True)
        self.transposed_conv_1 = nn.ConvTranspose3d(
            in_channels=32,
            out_channels=32,
            kernel_size=(3, 2, 2),
            stride=(1, 2, 2),
            padding=(1, 0, 0),
            output_padding=0,
        )
        self.red3d_2 = ResnetBlock(in_n_filters=32, n_filters=16, kernel_size=(3, 3, 3), valid=True)
        self.transposed_conv_2 = nn.ConvTranspose3d(
            in_channels=16,
            out_channels=16,
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
        self.out = ValidConv3D(16, 1, (3, 3, 3), initialization=init)

        if final_activation == "sigmoid":
            self.final_activation = torch.nn.Sigmoid()
        elif final_activation is not None:
            raise NotImplementedError(final_activation)
        else:
            self.final_activation = None

    def forward(self, tensors):
        x = tensors[self.input_name]
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

        tensors[self.prediction_name] = out
        return tensors

    @classmethod
    def get_scaling(cls, ipt_shape: Optional[Tuple[int, int]] = None) -> Tuple[float, float]:
        return 4.0, 4.0

    @classmethod
    def get_shrinkage(cls, ipt_shape: Optional[Tuple[int, int]] = None) -> Tuple[int, int]:
        return 13, 13

    def get_output_shape(self, ipt_shape: Tuple[int, int]):
        return tuple([i * sc - 2 * sr for i, sc, sr in zip(ipt_shape, self.get_scaling(), self.get_shrinkage())])


if __name__ == "__main__":
    m = M16(input_name="dummy", prediction_name="pred", nnum=19, z_out=49, final_activation=None)
    m = m.cuda()
    dummy = {"dummy": torch.ones(1, 361, 49, 74).cuda()}
    out = m(dummy)["pred"]
    print(out.shape)
