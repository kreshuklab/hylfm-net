import collections
import inspect
from enum import Enum
from functools import partial
from typing import Optional, List, Sequence, Tuple, Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


import torch.nn as nn

from hylfm.model.conv_layers import Conv2D, ResnetBlock, ValidConv3D
from inferno.extensions.initializers import Constant, Initialization


class HyLFM_Net(nn.Module):
    class InitName(str, Enum):
        uniform = "uniform_"
        normal = "normal_"
        constant = "constant_"
        eye = "eye_"
        dirac = "dirac_"
        xavier_uniform = "xavier_uniform_"
        xavier_normal = "xavier_normal_"
        kaiming_uniform = "kaiming_uniform_"
        kaiming_normal = "kaiming_normal_"
        orthogonal = "orthogonal_"
        sparse = "sparse_"

    def __init__(
        self,
        *,
        z_out: int,
        nnum: int,
        kernel2d: int = 3,
        conv_per_block2d: int = 2,
        c_res2d: Sequence[Union[int, str]] = (488, 488, "u244", 244),
        last_kernel2d: int = 1,
        c_in_3d: int = 7,
        kernel3d: int = 3,
        conv_per_block3d: int = 2,
        c_res3d: Sequence[str] = (7, "u7", 7, 7),
        init_fn: Union[InitName, str] = InitName.xavier_uniform,
        final_activation: Optional[Literal["sigmoid"]] = None,
    ):
        super().__init__()
        if isinstance(init_fn, str):
            init_fn = getattr(self.InitName, init_fn)

        init_fn = getattr(nn.init, init_fn.value)
        self.c_res2d = list(c_res2d)
        self.c_res3d = list(c_res3d)
        c_res3d = c_res3d
        self.nnum = nnum
        self.z_out = z_out
        if kernel3d != 3:
            raise NotImplementedError("z_out expansion for other res3d kernel")

        dz = 2 * conv_per_block3d * (kernel3d // 2)
        for c in c_res3d:
            if isinstance(c, int) or not c.startswith("u"):
                z_out += dz

        # z_out += 4 * (len(c_res3d) - 2 * sum([layer == "u" for layer in c_res3d]))  # add z_out for valid 3d convs

        assert c_res2d[-1] != "u", "missing # output channels for upsampling in 'c_res2d'"
        assert c_res3d[-1] != "u", "missing # output channels for upsampling in 'c_res3d'"

        res2d = []
        c_in = nnum ** 2
        c_out = c_in
        for i in range(len(c_res2d)):
            if not isinstance(c_res2d[i], int) and c_res2d[i].startswith("u"):
                c_out = int(c_res2d[i][1:])
                res2d.append(
                    nn.ConvTranspose2d(
                        in_channels=c_in, out_channels=c_out, kernel_size=2, stride=2, padding=0, output_padding=0
                    )
                )
            else:
                c_out = int(c_res2d[i])
                res2d.append(
                    ResnetBlock(
                        in_n_filters=c_in,
                        n_filters=c_out,
                        kernel_size=(kernel2d, kernel2d),
                        valid=False,
                        conv_per_block=conv_per_block2d,
                    )
                )

            c_in = c_out

        self.res2d = nn.Sequential(*res2d)

        if "gain" in inspect.signature(init_fn).parameters:
            init_fn_conv2d = partial(init_fn, gain=nn.init.calculate_gain("relu"))
        else:
            init_fn_conv2d = init_fn

        init = Initialization(weight_initializer=init_fn_conv2d, bias_initializer=Constant(0.0))
        self.conv2d = Conv2D(c_out, z_out * c_in_3d, last_kernel2d, activation="ReLU", initialization=init)

        self.c2z = lambda ipt, ip3=c_in_3d: ipt.view(ipt.shape[0], ip3, z_out, *ipt.shape[2:])

        res3d = []
        c_in = c_in_3d
        c_out = c_in
        for i in range(len(c_res3d)):
            if not isinstance(c_res3d[i], int) and c_res3d[i].startswith("u"):
                c_out = int(c_res3d[i][1:])
                res3d.append(
                    nn.ConvTranspose3d(
                        in_channels=c_in,
                        out_channels=c_out,
                        kernel_size=(3, 2, 2),
                        stride=(1, 2, 2),
                        padding=(1, 0, 0),
                        output_padding=0,
                    )
                )
            else:
                c_out = int(c_res3d[i])
                res3d.append(
                    ResnetBlock(
                        in_n_filters=c_in,
                        n_filters=c_out,
                        kernel_size=(kernel3d, kernel3d, kernel3d),
                        valid=True,
                        conv_per_block=conv_per_block3d,
                    )
                )

            c_in = c_out

        self.res3d = nn.Sequential(*res3d)

        if "gain" in inspect.signature(init_fn).parameters:
            init_fn_conv3d = partial(init_fn, gain=nn.init.calculate_gain("linear"))
        else:
            init_fn_conv3d = init_fn

        init = Initialization(weight_initializer=init_fn_conv3d, bias_initializer=Constant(0.0))
        self.conv3d = ValidConv3D(c_out, 1, (1, 1, 1), initialization=init)

        if final_activation is None:
            self.final_activation = None
        elif final_activation == "sigmoid":
            self.final_activation = nn.Sigmoid()
        else:
            raise NotImplementedError(final_activation)

    def forward(self, x):
        x = self.res2d(x)
        x = self.conv2d(x)
        x = self.c2z(x)
        x = self.res3d(x)
        x = self.conv3d(x)

        if self.final_activation is not None:
            x = self.final_activation(x)

        return x

    def load_state_dict_from_larger_z_out(self, state, z_out_start: int, z_out_end: int):
        converted = collections.OrderedDict()
        first_3d_channels = self.c_res3d[0][0]
        expect_diffs_in = ["conv2d.conv.weight", "conv2d.conv.bias"]
        diff_idx = 0
        for (sk, sv), (bk, bv) in zip(self.state_dict().items(), state.items()):
            assert sk == bk
            if sv.shape == bv.shape:
                converted[sk] = bv
            else:
                assert sk == expect_diffs_in[diff_idx]
                diff_idx += 1
                z_valid_offset = (sv.shape[0] / first_3d_channels - self.z_out) / 2
                assert z_valid_offset == int(z_valid_offset)
                z_valid_offset = int(z_valid_offset)
                # print(z_valid_offset)
                converted_v = bv[z_out_start * first_3d_channels : (2 * z_valid_offset + z_out_end) * first_3d_channels]
                # print(sk, sv.shape[0] / first_3d_channels, bv.shape[0] / first_3d_channels)
                # print(sv.shape, converted_v.shape, bv.shape)
                assert converted_v.shape == sv.shape
                converted[sk] = converted_v
        self.load_state_dict(converted)

    def get_scale(self, ipt_shape: Optional[Tuple[int, int]] = None) -> float:
        s = max(1, 2 * sum(isinstance(res2d, str) and res2d.startswith("u") for res2d in self.c_res2d)) * max(
            1, 2 * sum(isinstance(res3d, str) and res3d.startswith("u") for res3d in self.c_res3d)
        )
        return s

    def get_shrink(self, ipt_shape: Optional[Tuple[int, int]] = None) -> int:
        s = 0
        for res in self.c_res3d:
            if isinstance(res, str) and res.startswith("u"):
                s *= 2
            else:
                s += 2

        return s

    def get_output_shape(self, ipt_shape: Tuple[int, int]) -> Tuple[int, int, int]:
        scale = self.get_scaling(ipt_shape)
        shrink = self.get_shrink(ipt_shape)
        return (self.z_out,) + tuple(i * scale - 2 * shrink for i in ipt_shape)
