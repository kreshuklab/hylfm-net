from typing import Tuple, Optional
import logging
import torch.nn
import torch.nn as nn

from lnet.models.base import LnetModel

logger = logging.getLogger(__name__)


class M20(LnetModel):
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
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.bias = bias
        self.dilation = dilation
        self.tconv = nn.ConvTranspose2d(
            in_channels=nnum ** 2,
            out_channels=z_out,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=1,
            bias=bias,
            dilation=dilation,
        )

        if final_activation == "sigmoid":
            self.final_activation = torch.nn.Sigmoid()
        elif final_activation is not None:
            raise NotImplementedError(final_activation)
        else:
            self.final_activation = None

    def forward(self, x):
        out = self.tconv(x)
        if self.final_activation is not None:
            out = self.final_activation(out)

        return out

    def get_scaling(self, ipt_shape: Optional[Tuple[int, int]] = None) -> Tuple[float, float]:
        shrinkage = self.get_shrinkage(ipt_shape)
        out_shape = self.get_output_shape(ipt_shape)
        return tuple((o + 2 * s) / i for o, s, i in zip(out_shape, shrinkage, ipt_shape))

    def get_shrinkage(self, ipt_shape: Optional[Tuple[int, int]] = None) -> Tuple[int, int]:
        if self.output_padding != 0:
            raise NotImplementedError

        return self.padding, self.padding

    def get_output_shape(self, ipt_shape: Tuple[int, int]) -> Tuple[int, int]:
        return tuple(
            (i - 1) * self.stride - 2 * self.padding + self.dilation * (self.kernel_size - 1) + self.output_padding + 1
            for i in ipt_shape
        )


if __name__ == "__main__":
    ipt = torch.ones(1, 19**2, 10, 20)
    m20 = M20(z_out=7, nnum=19, kernel_size=4, padding=4, )
    print('shrink', m20.get_shrinkage())
    print('scale', m20.get_scaling(ipt.shape[-2:]))
    print(m20(ipt).shape)
