from functools import partial

import torch.nn
from inferno.extensions.initializers import Initialization, Constant
from typing import Tuple, Optional
import torch.nn as nn
from lnet.models.lnet import LnetModel

from lnet.models.layers.conv_layers import Conv2D, ValidConv2D, ValidConv3D, ResnetBlock
from lnet.models.layers.structural_layers import C2Z


class M13(LnetModel):
    def __init__(self, z_out: int, nnum: int, final_activation: Optional[str] = None):
        super().__init__()
        inplanes = nnum ** 2
        planes = 64
        z_valid_cut = 14
        z_out += z_valid_cut
        self.seq = torch.nn.Sequential()
        self.seq.add_module("res2d-1", ResnetBlock(in_n_filters=inplanes, n_filters=planes, valid=False))
        self.seq.add_module("res2d-2", ResnetBlock(in_n_filters=planes, n_filters=planes, valid=False))
        self.seq.add_module("res2d-3", ResnetBlock(in_n_filters=planes, n_filters=planes, valid=False))

        inplanes = planes
        planes = z_out
        init = partial(
            Initialization,
            weight_initializer=partial(nn.init.xavier_uniform_, gain=nn.init.calculate_gain("relu")),
            bias_initializer=Constant(0.0),
        )
        self.seq.add_module("conv2", Conv2D(inplanes, planes, (3, 3), activation="ReLU", initialization=init))

        c2z = C2Z(z_out)
        inplanes = c2z.get_c_out(planes)
        planes = 64
        self.seq.add_module("C2Z", c2z)
        self.seq.add_module(
            "red3d-1", ResnetBlock(in_n_filters=inplanes, n_filters=planes, kernel_size=(3, 3, 3), valid=True)
        )
        self.seq.add_module(
            "transposed-conv-1",
            nn.ConvTranspose3d(
                in_channels=planes,
                out_channels=planes,
                kernel_size=(3, 2, 2),
                stride=(1, 2, 2),
                padding=(1, 0, 0),
                output_padding=0,
            ),
        )
        self.seq.add_module(
            "red3d-2", ResnetBlock(in_n_filters=planes, n_filters=planes, kernel_size=(3, 3, 3), valid=True)
        )
        self.seq.add_module(
            "transposed-conv-2",
            nn.ConvTranspose3d(
                in_channels=planes,
                out_channels=planes,
                kernel_size=(3, 2, 2),
                stride=(1, 2, 2),
                padding=(1, 0, 0),
                output_padding=0,
            ),
        )
        in_planes = planes
        planes = 16
        self.seq.add_module(
            "red3d-3", ResnetBlock(in_n_filters=in_planes, n_filters=planes, kernel_size=(3, 3, 3), valid=True)
        )
        self.seq.add_module(
            "transposed-conv-3",
            nn.ConvTranspose3d(
                in_channels=planes,
                out_channels=planes,
                kernel_size=(3, 2, 2),
                stride=(1, 2, 2),
                padding=(1, 0, 0),
                output_padding=0,
            ),
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

    def forward(self, input):
        intermediate = self.seq.forward(input)
        out = self.out(intermediate)

        if self.final_activation is not None:
            out = self.final_activation(out)

        return out

    @classmethod
    def get_scaling(cls, ipt_shape: Optional[Tuple[int, int]] = None) -> Tuple[float, float]:
        return 8.0, 8.0

    @classmethod
    def get_shrinkage(cls, ipt_shape: Optional[Tuple[int, int]] = None) -> Tuple[int, int]:
        return 29, 29

    def get_output_shape(self, ipt_shape: Tuple[int, int]):
        return tuple([i * sc - 2 * sr for i, sc, sr in zip(ipt_shape, self.get_scaling(), self.get_shrinkage())])
