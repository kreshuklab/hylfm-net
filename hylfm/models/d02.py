import inspect
import logging
from functools import partial
from typing import Callable, Dict, Optional, Sequence, Tuple, Union

import torch.nn
import torch.nn as nn
from inferno.extensions.initializers import Constant, Initialization

from hylfm import registration
from hylfm.models.base import LnetModel
from hylfm.models.layers.conv_layers import ResnetBlock
from hylfm.models.layers.dense_block import DenseBlock
from hylfm.models.layers.transition import Transition2D, Transition3D

logger = logging.getLogger(__name__)


class D02(LnetModel):
    def __init__(
        self,
        z_out: int,
        nnum: int,
        affine_transform_classes: Dict[str, str],
        interpolation_order: int,
        grid_sampling_scale: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        final_activation: Optional[str] = None,
        block_config_2d=(6, 12, "16u", 16),
        inplanes_3d: int = 8,
        n_res3d: Sequence[Sequence[int]] = ((8, 7), (7,)),
        init_fn: Callable = nn.init.xavier_uniform_,
        growth_rate: Union[str, int] = "z_out",
        bn_size=2,
        drop_rate=0,
        batch_norm: bool = False,
    ):
        assert interpolation_order in [0, 2]
        # assert len(n_res3d) >= 1, n_res3d
        super().__init__()
        self.grid_sampling_scale = grid_sampling_scale
        assert int(z_out * self.grid_sampling_scale[0]) == z_out * self.grid_sampling_scale[0]
        # self.n_dense2d = n_dense2d
        # n_dense2d = [nnum ** 2] + list(n_dense2d)
        self.n_res3d = n_res3d
        self.z_out = z_out
        z_out += 4 * len(n_res3d)
        self.block_config_2d = block_config_2d
        # if n_res3d[-1][-1] != 1:
        #     z_out += 2  # add z_out for valid 3d convs

        if growth_rate == "z_out":
            growth_rate = z_out
        else:
            assert isinstance(growth_rate, int)

        self.dense2d = nn.Sequential()
        # Each denseblock
        num_features = nnum ** 2
        for i, num_layers in enumerate(block_config_2d):
            if isinstance(num_layers, str):
                assert num_layers.endswith("u")
                assert i != len(block_config_2d) - 1
                num_layers = int(num_layers[:-1])
                upsample = True
            else:
                upsample = False

            block = DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                batch_norm=batch_norm,
            )
            self.dense2d.add_module("denseblock%d" % (i + 1), block)
            num_features = num_features + num_layers * growth_rate

            if upsample:
                trans = nn.ConvTranspose2d(
                    in_channels=num_features,
                    out_channels=num_features // 2,
                    kernel_size=(2, 2),
                    stride=(2, 2),
                    padding=(0, 0),
                    output_padding=0,
                )
            elif i != len(block_config_2d) - 1:
                trans = Transition2D(num_input_features=num_features, num_output_features=num_features // 2)
            else:
                trans = None

            if trans is not None:
                self.dense2d.add_module("transition%d" % (i + 1), trans)
                num_features //= 2

        self.dense2d.add_module(
            "transition_zout*inplances3d",
            Transition2D(num_input_features=num_features, num_output_features=z_out * inplanes_3d),
        )

        self.c2z = lambda ipt, ip3=inplanes_3d: ipt.view(ipt.shape[0], ip3, z_out, *ipt.shape[2:])

        res3d = []
        for n in n_res3d:
            res3d.append(ResnetBlock(in_n_filters=inplanes_3d, n_filters=n[0], kernel_size=(3, 3, 3), valid=True))
            if len(n) > 1:
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

        init = partial(Initialization, weight_initializer=init_fn_conv3d, bias_initializer=Constant(0.0))
        if n_res3d[-1][-1] != 1:
            self.res3d.add_module("out_transition", Transition3D(res3d_out_channels, 1, batch_norm=batch_norm))

        # self.conv3d = ValidConv3D(res3d_out_channels, 1, (3, 3, 3), initialization=init)

        # todo: make learnable with ModuleDict

        self.affine_transforms = {
            in_shape_for_at: getattr(registration, at_class)(
                order=interpolation_order, trf_out_zoom=grid_sampling_scale
            )
            for in_shape_for_at, at_class in affine_transform_classes.items()
        }
        self.z_dims = {
            in_shape: at.ls_shape[0] - at.lf2ls_crop[0][0] - at.lf2ls_crop[0][1]
            for in_shape, at in self.affine_transforms.items()
        }

        if final_activation == "sigmoid":
            self.final_activation = torch.nn.Sigmoid()
        elif final_activation is not None:
            raise NotImplementedError(final_activation)
        else:
            self.final_activation = None

    def forward(self, x, z_slices: Optional[Sequence[int]] = None):
        in_shape = ",".join(str(s) for s in x.shape[1:])
        x = self.dense2d(x)
        x = self.c2z(x)
        x = self.res3d(x)

        if z_slices is None:
            out_shape = tuple(int(s * g) for s, g in zip(x.shape[2:], self.grid_sampling_scale))
        else:
            z_dim = int(self.z_dims[in_shape] * self.grid_sampling_scale[0])
            out_shape = (z_dim,) + tuple(int(s * g) for s, g in zip(x.shape[3:], self.grid_sampling_scale[1:]))

        x = self.affine_transforms[in_shape](x, output_shape=out_shape, z_slices=z_slices)

        if self.final_activation is not None:
            x = self.final_activation(x)

        return x

    def get_scaling(self, ipt_shape: Optional[Tuple[int, int]] = None) -> Tuple[float, float]:
        s = max(1, 2 * sum(isinstance(block2d, str) and "u" in block2d for block2d in self.block_config_2d)) * max(
            1, 2 * sum(len(up3d) == 2 for up3d in self.n_res3d)
        )
        return s, s

    def get_shrinkage(self, ipt_shape: Optional[Tuple[int, int]] = None) -> Tuple[int, int]:
        s = 0
        for res3d in self.n_res3d:
            s += 2
            if len(res3d) > 1:
                s *= 2

        # grid sampling scale
        sfloat = (s * self.grid_sampling_scale[1], s * self.grid_sampling_scale[2])
        s = (int(sfloat[0]), int(sfloat[1]))
        assert s == sfloat

        return s

    def get_output_shape(self, ipt_shape: Tuple[int, int]) -> Tuple[int, int, int]:
        return (int(self.z_out * self.grid_sampling_scale[0]),) + tuple(
            i * sc - 2 * sr for i, sc, sr in zip(ipt_shape, self.get_scaling(), self.get_shrinkage())
        )


def try_static():
    import yaml

    import matplotlib.pyplot as plt
    from config.__init__ import DataConfig, DataCategory

    from config.__init__ import ModelConfig

    model_config = ModelConfig.load(
        D02.__name__,
        z_out=79,
        nnum=19,
        precision="float",
        kwargs=yaml.safe_load(
            """
affine_transform_classes:
    361,67,77: Heart_tightCrop_Transform
    361,77,67: Heart_tightCrop_Transform
    361,66,77: Heart_tightCrop_Transform
    361,77,66: Heart_tightCrop_Transform
    361,62,93: staticHeartFOV_Transform
    361,93,62: staticHeartFOV_Transform
interpolation_order: 2
grid_sampling_scale: [1.0, 1.0, 1.0]
block_config_2d: [6, 12, 16u, 12]
inplanes_3d:  8
n_res3d: [[8, 7], [7, 1]]
growth_rate: z_out
batch_norm: true
"""
        ),
    )
    data_config = DataConfig.load(
        model_config=model_config,
        category=DataCategory.test,
        entries=yaml.safe_load(
            """
fish2_20191209.t0815_static_affine: {indices: null, interpolation_order: 2, save: false}
"""
        ),
        default_batch_size=1,
        default_transforms=yaml.safe_load(
            """
- {name: norm01, kwargs: {apply_to: 0, percentile_min: 5.0, percentile_max: 99.8}}
- {name: norm01, kwargs: {apply_to: 1, percentile_min: 5.0, percentile_max: 99.99}}
- Lightfield2Channel
"""
        ),
    )
    m = model_config.model
    device = torch.device("cuda")
    m = m.to(device)

    if model_config.checkpoint is not None:
        state = torch.load(model_config.checkpoint, map_location=device)
        m.load_state_dict(state, strict=False)

    loader = data_config.data_loader
    # ipt = torch.rand(1, nnum ** 2, 5, 5)
    ipt, tgt = next(iter(loader))
    ipt, tgt = ipt.to(device), tgt.to(device)
    print("get_scaling", m.get_scaling(ipt.shape[2:]))
    print("get_shrinkage", m.get_shrinkage(ipt.shape[2:]))
    print("get_output_shape()", m.get_output_shape(ipt.shape[2:]))
    print("ipt", ipt.shape)
    print("tgt", tgt.shape)
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

    print("done")


def try_dynamic():
    import yaml

    import matplotlib.pyplot as plt
    from config.__init__ import DataConfig, DataCategory

    from config.__init__ import ModelConfig

    model_config = ModelConfig.load(
        D02.__name__,
        z_out=79,
        nnum=19,
        precision="float",
        kwargs=yaml.safe_load(
            """
affine_transform_classes:
  361,67,66: fast_cropped_8ms_Transform
  361,66,67: fast_cropped_8ms_Transform
interpolation_order: 2
"""
        ),
    )
    data_config = DataConfig.load(
        model_config=model_config,
        category=DataCategory.test,
        entries=yaml.safe_load(
            """
fish2_20191209_dynamic.t0402c11p100a: {indices: null, interpolation_order: 2, save: false}
"""
        ),
        default_batch_size=1,
        default_transforms=yaml.safe_load(
            """
- {name: norm01, kwargs: {apply_to: 0, percentile_min: 5.0, percentile_max: 99.8}}
- {name: norm01, kwargs: {apply_to: 1, percentile_min: 5.0, percentile_max: 99.99}}
- Lightfield2Channel
"""
        ),
    )
    m = model_config.model
    device = torch.device("cuda")
    m = m.to(device)

    if model_config.checkpoint is not None:
        state = torch.load(model_config.checkpoint, map_location=device)
        m.load_state_dict(state, strict=False)

    loader = data_config.data_loader
    ipt, tgt, z_slice = next(iter(loader))
    ipt, tgt = ipt.to(device), tgt.to(device)
    print("get_scaling", m.get_scaling(ipt.shape[2:]))
    print("get_shrinkage", m.get_shrinkage(ipt.shape[2:]))
    print("get_output_shape()", m.get_output_shape(ipt.shape[2:]))
    print("ipt", ipt.shape)
    print("tgt", tgt.shape)
    plt.imshow(tgt[0, 0].detach().cpu().numpy())
    plt.title("tgt")
    plt.show()
    # print("scale", m.get_scaling(ipt.shape[2:]))
    # print("out", m.get_output_shape(ipt.shape[2:]))
    # print("shrink", m.get_shrinkage(ipt.shape[2:]))
    # print("len 3d", len(m.res3d))

    out = m(ipt, z_slices=[z_slice])
    print("out", out.shape)
    plt.imshow(out[0, 0].detach().cpu().numpy())
    plt.title(f"out {z_slice}")
    plt.show()

    print("done")


if __name__ == "__main__":
    try_static()
    # try_dynamic()
    print("max mem alloc", torch.cuda.max_memory_allocated() / 2 ** 30)
    print("max mem cache", torch.cuda.max_memory_cached() / 2 ** 30)