import inspect
from typing import Tuple, Optional, Sequence, Callable, Dict
import logging

import numpy
import torch.nn
import torch.nn as nn

from functools import partial

from inferno.extensions.initializers import Initialization, Constant

from lnet.config.dataset import registration

from lnet.models.layers.conv_layers import Conv2D, ValidConv2D, ValidConv3D, ResnetBlock
from lnet.models.base import LnetModel

logger = logging.getLogger(__name__)


class A01(LnetModel):
    def __init__(
        self,
        z_out: int,
        nnum: int,
        affine_transform_classes: Dict[str, str],
        interpolation_order: int,
        grid_sampling_scale: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        final_activation: Optional[str] = None,
        n_res2d: Sequence[int] = (256, 128, 98),
        inplanes_3d: int = 2,
        n_res3d: Sequence[Sequence[int]] = ((32, 32), (16, 16)),
        init_fn: Callable = nn.init.xavier_uniform_,
    ):
        assert interpolation_order in [0, 2]
        # assert len(n_res3d) >= 1, n_res3d
        super().__init__()
        self.grid_sampling_scale = grid_sampling_scale
        assert int(z_out * self.grid_sampling_scale[0]) == z_out * self.grid_sampling_scale[0]
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

        if "gain" in inspect.signature(init_fn).parameters:
            init_fn_conv2d = partial(init_fn, gain=nn.init.calculate_gain("relu"))
        else:
            init_fn_conv2d = init_fn

        init = partial(Initialization, weight_initializer=init_fn_conv2d, bias_initializer=Constant(0.0))
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
        if n_res3d:
            res3d_out_channels = n_res3d[-1][-1]
        else:
            res3d_out_channels = inplanes_3d

        if "gain" in inspect.signature(init_fn).parameters:
            init_fn_conv3d = partial(init_fn, gain=nn.init.calculate_gain("linear"))
        else:
            init_fn_conv3d = init_fn

        init = partial(Initialization, weight_initializer=init_fn_conv3d, bias_initializer=Constant(0.0))
        self.conv3d = ValidConv3D(res3d_out_channels, 1, (3, 3, 3), initialization=init)

        # todo: make learnable with ModuleDict

        self.affine_transforms = {
            in_shape_for_at: getattr(registration, at_class)(order=interpolation_order)
            for in_shape_for_at, at_class in affine_transform_classes.items()
        }

        if final_activation == "sigmoid":
            self.final_activation = torch.nn.Sigmoid()
        elif final_activation is not None:
            raise NotImplementedError(final_activation)
        else:
            self.final_activation = None

    def forward(self, x):
        in_shape = ",".join(x.shape)
        x = self.res2d(x)
        x = self.conv2d(x)
        x = self.c2z(x)
        x = self.res3d(x)
        x = self.conv3d(x) + 0.1

        x = self.affine_transforms[in_shape](
            x, output_shape=tuple(int(s * g) for s, g in zip(x.shape[2:], self.grid_sampling_scale))
        )

        if self.final_activation is not None:
            x = self.final_activation(x)

        return x

    def get_scaling(self, ipt_shape: Optional[Tuple[int, int]] = None) -> Tuple[float, float]:
        s = max(1, 2 * len(self.n_res3d))
        return s, s

    def get_shrinkage(self, ipt_shape: Optional[Tuple[int, int]] = None) -> Tuple[int, int]:
        s = 0
        for _ in self.n_res3d:
            s += 2
            s *= 2

        s += 1  # 3d valid conv

        # grid sampling scale
        sfloat = (s * self.grid_sampling_scale[1], s * self.grid_sampling_scale[2])
        s = (int(sfloat[0]), int(sfloat[1]))
        assert s == sfloat

        return s

    def get_output_shape(self, ipt_shape: Tuple[int, int]) -> Tuple[int, int, int]:
        return (int(self.z_out * self.grid_sampling_scale[0]),) + tuple(
            i * sc - 2 * sr for i, sc, sr in zip(ipt_shape, self.get_scaling(), self.get_shrinkage())
        )


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from lnet.config.data import DataConfig, DataCategory

    from lnet.config.model import ModelConfig

    interpolation_order = 0
    model_config = ModelConfig.load(
        "A01",
        z_out=49,
        nnum=19,
        precision="float",
        checkpoint="/g/kreshuk/beuttenm/repos/lnet/logs/fish/fish2_20191208_0815_static/20-01-24_07-47-11/models/v0_model_260.pth",
        kwargs={
            "n_res2d": [128, 64],  # [],
            "inplanes_3d": 32,  # 1,
            "n_res3d": [[32, 16], [8, 4]],  # [],
            # "init_fn": nn.init.dirac_,
            "affine_transform_class": "Heart_tightCrop_Transform",
            "interpolation_order": interpolation_order,
            # "grid_sampling_scale: (1.5, 2, 2),
        },
    )
    data_config = DataConfig.load(
        model_config=model_config,
        category=DataCategory.test,
        entries={
            "fish2_20191209_0815_static_affine": {
                "indices": None,
                "interpolation_order": 2,
                # "affine_transformation": "default",
            }
        },
        default_batch_size=1,
        default_transforms=[
            {"name": "norm01", "kwargs": {"apply_to": 0, "percentile_min": 5.0, "percentile_max": 99.8}},
            {"name": "norm01", "kwargs": {"apply_to": 1, "percentile_min": 5.0, "percentile_max": 99.99}},
            "Lightfield2Channel",
        ],
    )
    m = model_config.model
    # m.cuda()
    if model_config.checkpoint is not None:
        state = torch.load(model_config.checkpoint, map_location=torch.device("cpu"))
        m.load_state_dict(state, strict=False)

    loader = data_config.data_loader
    # ipt = torch.rand(1, nnum ** 2, 5, 5)
    ipt, tgt = next(iter(loader))
    print("ipt", ipt.shape, "tgt", tgt.shape)
    plt.imshow(tgt[0, 0].detach().cpu().numpy().max(axis=0))
    plt.title("tgt")
    plt.show()
    plt.imshow(tgt[0, 0].detach().cpu().numpy().max(axis=1))
    plt.title("tgt")
    plt.show()
    plt.imshow(tgt[0, 0].detach().cpu().numpy().max(axis=2))
    plt.title("tgt")
    plt.show()
    # print("scale", m.get_scaling(ipt.shape[2:]))
    # print("out", m.get_output_shape(ipt.shape[2:]))
    # print("shrink", m.get_shrinkage(ipt.shape[2:]))
    # print("len 3d", len(m.res3d))
    out = m(ipt)
    print("out", out.shape)
    plt.imshow(out[0, 0].detach().cpu().numpy().max(axis=0))
    plt.title("out")
    plt.show()
    plt.imshow(out[0, 0].detach().cpu().numpy().max(axis=1))
    plt.title("out")
    plt.show()
    plt.imshow(out[0, 0].detach().cpu().numpy().max(axis=2))
    plt.title("out")
    plt.show()
