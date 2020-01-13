from typing import Tuple, Optional, Sequence
import logging
import torch.nn
import torch.nn as nn


from lnet.models.layers.conv_layers import ResnetBlock
from lnet.models.base import LnetModel

logger = logging.getLogger(__name__)


class M21(LnetModel):
    def __init__(
        self,
        z_out: int,
        nnum: int,
        final_activation: Optional[str] = None,
        kernel_size: int = 8,
        stride: int = 4,
        padding: int = 4,
        output_padding: int = 0,
        bias: bool = True,
        dilation: int = 1,
        n_res2d: Sequence[int] = (256, 128),
        n_res3d: Sequence[int] = (32, 16),
    ):
        assert len(n_res3d) >= 1, n_res3d
        super().__init__()
        self.n_res2d = n_res2d
        n_res2d = [nnum**2] + list(n_res2d)
        self.n_res3d = n_res3d
        self.z_out = z_out
        z_out += 4 * len(n_res3d)  # add z_out for valid 3d convs
        n_res3d = list(n_res3d) + [1]

        self.res2d = nn.Sequential(
            *[
                ResnetBlock(in_n_filters=n_res2d[i], n_filters=n_res2d[i + 1], valid=False)
                for i in range(len(n_res2d) - 1)
            ]
        )

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.bias = bias
        self.dilation = dilation
        self.tconv = nn.ConvTranspose2d(
            in_channels=n_res2d[-1] if n_res2d else nnum ** 2,
            out_channels=n_res3d[0] * z_out if n_res3d else z_out,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=1,
            bias=bias,
            dilation=dilation,
        )

        self.c2z = lambda ipt: ipt.view(ipt.shape[0], n_res3d[0] if n_res3d else 1, z_out, *ipt.shape[2:])
        self.res3d = nn.Sequential(
            *[
                ResnetBlock(in_n_filters=n_res3d[i], n_filters=n_res3d[i + 1], kernel_size=(3, 3, 3), valid=True)
                for i in range(len(n_res3d) - 1)
            ]
        )

        if final_activation == "sigmoid":
            self.final_activation = torch.nn.Sigmoid()
        elif final_activation is not None:
            raise NotImplementedError(final_activation)
        else:
            self.final_activation = None

    def forward(self, x):
        # print('in', x.shape)
        x = self.res2d(x)
        # print('res2d', x.shape)
        x = self.tconv(x)
        # print('tconv', x.shape)
        x = self.c2z(x)
        # print('c2z', x.shape)
        x = self.res3d(x)
        # print('res3d', x.shape)

        if self.final_activation is not None:
            x = self.final_activation(x)

        return x

    def get_scaling(self, ipt_shape: Optional[Tuple[int, int]] = None) -> Tuple[float, float]:
        if self.dilation != 1:
            raise NotImplementedError

        return self.stride, self.stride

    def get_shrinkage(self, ipt_shape: Optional[Tuple[int, int]]) -> Tuple[int, int]:
        if self.output_padding != 0:
            raise NotImplementedError

        scale = self.get_scaling(ipt_shape)
        upscaled = [i * s for i, s in zip(ipt_shape, scale)]
        out = self.get_output_shape(ipt_shape)[1:]
        diff = [u - o for u, o in zip(upscaled, out)]
        assert all(d % 2 == 0 for d in diff), diff
        # assert all(d //2 == self.padding for d in diff), (diff, self.padding)

        return tuple(d // 2 for d in diff)

    def get_output_shape(self, ipt_shape: Tuple[int, int]) -> Tuple[int, int, int]:
        out2d = tuple(
            (i - 1) * self.stride - 2 * self.padding + self.dilation * (self.kernel_size - 1) + self.output_padding + 1 - 4 * len(self.n_res3d)
            for i in ipt_shape
        )
        return (self.z_out, ) + out2d


if __name__ == "__main__":
    ipt = torch.ones(1, 19 ** 2, 10, 20)
    m = M21(z_out=7, nnum=19, kernel_size=8, padding=4, stride=4)
    print("scale", m.get_scaling(ipt.shape[-2:]))
    print('out', m.get_output_shape(ipt.shape[-2:]))
    print("shrink", m.get_shrinkage(ipt.shape[-2:]))
    print('len 3d', len(m.res3d))
    print(m(ipt).shape)
