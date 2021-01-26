import inspect
from functools import partial
from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple

import torch
from torch import nn

from hylfm.model.conv_layers import Conv2D, ResnetBlock, ValidConv3D
from inferno.extensions.initializers import Constant, Initialization


class A04(nn.Module):
    def __init__(
        self,
        *,
        z_out: int,
        nnum: int,
        scale: int,
        shrink: int,
        final_activation: Optional[str] = None,
        n_res2d: Sequence[int] = (976, 976, "u", 488, 488, "u", 244, 244),
        inplanes_3d: int = 7,
        n_res3d: Sequence[Sequence[int]] = ((7, 7), (7,), (7,)),
        init_fn: Callable = nn.init.xavier_uniform_,
        last_2d_kernel: Tuple[int, int] = (1, 1),
    ):
        # assert len(n_res3d) >= 1, n_res3d
        super().__init__()
        self.n_res2d = n_res2d
        n_res2d = [nnum ** 2] + list(n_res2d)
        self.n_res3d = n_res3d
        self.z_out = z_out
        z_out += 4 * len(n_res3d)  # add z_out for valid 3d convs
        print("z_out", z_out)

        res2d = []
        for i in range(len(n_res2d) - 1):
            if n_res2d[i] == "u":
                assert i > 0
                assert n_res2d[i + 1] != "u"
                res2d.append(
                    nn.ConvTranspose2d(
                        in_channels=n_res2d[i - 1],
                        out_channels=n_res2d[i + 1],
                        kernel_size=2,
                        stride=2,
                        padding=0,
                        output_padding=0,
                    )
                )
            elif n_res2d[i + 1] == "u":
                continue
            else:
                assert isinstance(n_res2d[i], int) and isinstance(n_res2d[i + 1], int), (n_res2d[i], n_res2d[i + 1])
                res2d.append(ResnetBlock(in_n_filters=n_res2d[i], n_filters=n_res2d[i + 1], valid=False))
                print('in', n_res2d[i], 'out', n_res2d[i + 1])

        self.res2d = nn.Sequential(*res2d)

        if "gain" in inspect.signature(init_fn).parameters:
            init_fn_conv2d = partial(init_fn, gain=nn.init.calculate_gain("relu"))
        else:
            init_fn_conv2d = init_fn

        init = Initialization(weight_initializer=init_fn_conv2d, bias_initializer=Constant(0.0))
        self.conv2d = Conv2D(n_res2d[-1], z_out * inplanes_3d, last_2d_kernel, activation="ReLU", initialization=init)

        self.c2z = lambda ipt, ip3=inplanes_3d: ipt.view(ipt.shape[0], ip3, z_out, *ipt.shape[2:])

        res3d = []
        for n in n_res3d:
            print('3d: in', inplanes_3d, 'out', n[0])
            res3d.append(ResnetBlock(in_n_filters=inplanes_3d, n_filters=n[0], kernel_size=(3, 3, 3), valid=True))
            if len(n) == 2:
                print('up: in', n[0], 'out', n[1])
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
            else:
                inplanes_3d = n[0]

        self.res3d = nn.Sequential(*res3d)
        if n_res3d:
            res3d_out_channels = n_res3d[-1][-1]
        else:
            res3d_out_channels = inplanes_3d

        if "gain" in inspect.signature(init_fn).parameters:
            init_fn_conv3d = partial(init_fn, gain=nn.init.calculate_gain("linear"))
        else:
            init_fn_conv3d = init_fn

        init = Initialization(weight_initializer=init_fn_conv3d, bias_initializer=Constant(0.0))
        self.conv3d = ValidConv3D(res3d_out_channels, 1, (1, 1, 1), initialization=init)

        if final_activation is None:
            self.final_activation = None
        elif final_activation == "Sigmoid":
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


if __name__ == "__main__":
    # checkpoint = Path(r"C:\Users\fbeut\Desktop\hylfm_stuff\old_checkpoints\v1_checkpoint_498_MS_SSIM=0.9710696664723483.pth")
    #
    # a = A04(nnum=19, z_out=49, n_res2d=[488, 488, "u", 244, 244], shrink=8, scale=4)
    # device = torch.device("cpu")
    # state = torch.load(str(checkpoint), map_location=device)["model"]
    # a.load_state_dict(state, strict=True)


    x = torch.zeros((1, 3, 60, 70, 80))
    # op = nn.ConvTranspose3d(
    #                     in_channels=3,
    #                     out_channels=4,
    #                     kernel_size=(3, 2, 2),
    #                     stride=(1, 2, 2),
    #                     padding=(1, 0, 0),
    #                     output_padding=0,
    #                 )

    op = ResnetBlock(3,
        3,
        kernel_size=(3, 3, 3),
        batch_norm=False,
        conv_per_block=2,
        valid= True,
        activation = "ReLU",
    )
    y = op(x)

    print(x.shape)
    print(y.shape)
