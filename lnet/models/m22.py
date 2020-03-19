import logging
from functools import partial
from typing import Optional, Sequence, Tuple

import torch.nn
import torch.nn as nn
from inferno.extensions.initializers import Constant, Initialization

from lnet.models.base import LnetModel
from lnet.models.layers.conv_layers import Conv2D, ResnetBlock, ValidConv3D

logger = logging.getLogger(__name__)


class M22(LnetModel):
    def __init__(
        self,
        z_out: int,
        nnum: int,
        final_activation: Optional[str] = None,
        n_res2d: Sequence[int] = (256, 128, 98),
        inplanes_3d: int = 2,
        n_res3d: Sequence[Sequence[int]] = ((32, 32), (16, 16)),
    ):
        assert len(n_res3d) >= 1, n_res3d
        super().__init__()
        self.n_res2d = n_res2d
        n_res2d = [nnum ** 2] + list(n_res2d)
        self.n_res3d = n_res3d
        self.z_out = z_out
        z_out += 4 * len(n_res3d) + 2  # add z_out for valid 3d convs

        self.res2d = nn.Sequential(
            *[
                ResnetBlock(in_n_filters=n_res2d[i], n_filters=n_res2d[i + 1], valid=False)
                for i in range(len(n_res2d) - 1)
            ]
        )

        init = partial(
            Initialization,
            weight_initializer=partial(nn.init.xavier_uniform_, gain=nn.init.calculate_gain("relu")),
            bias_initializer=Constant(0.0),
        )
        self.conv2d = Conv2D(n_res2d[-1], z_out * inplanes_3d, (3, 3), activation="ReLU", initialization=init)

        self.c2z = lambda ipt, ip3=inplanes_3d: ipt.view(ipt.shape[0], ip3, z_out, *ipt.shape[2:])

        res3d = []
        for n in n_res3d:
            res3d.append(ResnetBlock(in_n_filters=inplanes_3d, n_filters=n[0], kernel_size=(3, 3, 3), valid=True))
            res3d.append(
                nn.ConvTranspose3d(
                    in_channels=n[0],
                    out_channels=n[1],
                    kernel_size=(3, 2, 2),
                    stride=(1, 2, 2),
                    padding=(1, 0, 0),
                    output_padding=0,
                )
            )
            inplanes_3d = n[1]

        self.res3d = nn.Sequential(*res3d)
        init = partial(
            Initialization,
            weight_initializer=partial(nn.init.xavier_uniform_, gain=nn.init.calculate_gain("linear")),
            bias_initializer=Constant(0.0),
        )
        self.conv3d = ValidConv3D(n_res3d[-1][-1], 1, (3, 3, 3), initialization=init)

        if final_activation == "sigmoid":
            self.final_activation = torch.nn.Sigmoid()
        elif final_activation is not None:
            raise NotImplementedError(final_activation)
        else:
            self.final_activation = None

    def forward(self, x):
        x = self.res2d(x)
        x = self.conv2d(x)
        x = self.c2z(x)
        x = self.res3d(x)
        x = self.conv3d(x)

        if self.final_activation is not None:
            x = self.final_activation(x)

        return x

    def get_scaling(self, ipt_shape: Optional[Tuple[int, int]] = None) -> Tuple[float, float]:
        s = 2 * len(self.n_res3d)
        return s, s

    def get_shrinkage(self, ipt_shape: Optional[Tuple[int, int]] = None) -> Tuple[int, int]:
        s = 0
        for _ in self.n_res3d:
            s += 2
            s *= 2

        s += 1  # 3d valid conv

        return s, s

    def get_output_shape(self, ipt_shape: Tuple[int, int]):
        return tuple([i * sc - 2 * sr for i, sc, sr in zip(ipt_shape, self.get_scaling(), self.get_shrinkage())])


if __name__ == "__main__":
    ipt = torch.ones(1, 19 ** 2, 10, 20)
    m = M22(z_out=7, nnum=19, n_res2d=[256, 128], inplanes_3d=64, n_res3d=[[64, 32], [8, 4]])
    print("scale", m.get_scaling(ipt.shape[-2:]))
    print("out", m.get_output_shape(ipt.shape[-2:]))
    print("shrink", m.get_shrinkage(ipt.shape[-2:]))
    print("len 3d", len(m.res3d))
    print(m(ipt).shape)
