import inspect
import logging
import typing
from functools import partial
from typing import Callable, Optional, Sequence, Tuple

import torch.nn
import torch.nn as nn
from inferno.extensions.initializers import Constant, Initialization

from lnet.datasets import get_collate_fn
from lnet.datasets.base import TensorInfo
from lnet.datasets.beads import b01highc_0_ls
from lnet.models.base import LnetModel
from lnet.models.layers.conv_layers import Conv2D, ResnetBlock, ValidConv3D

logger = logging.getLogger(__name__)


class A04(LnetModel):
    def __init__(
        self,
        *,
        input_name: str,
        prediction_name: str,
        z_out: int,
        nnum: int,
        final_activation: Optional[str] = None,
        n_res2d: Sequence[int] = (976, 976, "u", 488, 488, "u", 244, 244),
        inplanes_3d: int = 7,
        n_res3d: Sequence[Sequence[int]] = ((7, 7), (7,), (7,)),
        init_fn: Callable = nn.init.xavier_uniform_,
    ):
        # assert len(n_res3d) >= 1, n_res3d
        super().__init__()
        self.input_name = input_name
        self.prediction_name = prediction_name
        self.n_res2d = n_res2d
        n_res2d = [nnum ** 2] + list(n_res2d)
        self.n_res3d = n_res3d
        self.z_out = z_out
        z_out += 4 * len(n_res3d)  # add z_out for valid 3d convs

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

        self.res2d = nn.Sequential(*res2d)

        if "gain" in inspect.signature(init_fn).parameters:
            init_fn_conv2d = partial(init_fn, gain=nn.init.calculate_gain("relu"))
        else:
            init_fn_conv2d = init_fn

        init = partial(Initialization, weight_initializer=init_fn_conv2d, bias_initializer=Constant(0.0))
        self.conv2d = Conv2D(n_res2d[-1], z_out * inplanes_3d, (1, 1), activation="ReLU", initialization=init)

        self.c2z = lambda ipt, ip3=inplanes_3d: ipt.view(ipt.shape[0], ip3, z_out, *ipt.shape[2:])

        res3d = []
        for n in n_res3d:
            res3d.append(ResnetBlock(in_n_filters=inplanes_3d, n_filters=n[0], kernel_size=(3, 3, 3), valid=True))
            if len(n) == 2:
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
        self.conv3d = ValidConv3D(res3d_out_channels, 1, (1, 1, 1), initialization=init)

        if final_activation == "sigmoid":
            self.final_activation = torch.nn.Sigmoid()
        elif final_activation is not None:
            raise NotImplementedError(final_activation)
        else:
            self.final_activation = None

    def forward(self, tensors: typing.OrderedDict[str, typing.Any]):
        x = tensors[self.input_name]
        x = self.res2d(x)
        x = self.conv2d(x)
        x = self.c2z(x)
        x = self.res3d(x)
        x = self.conv3d(x)

        if self.final_activation is not None:
            x = self.final_activation(x)

        tensors[self.prediction_name] = x
        return tensors

    def get_scaling(self, ipt_shape: Optional[Tuple[int, int]] = None) -> Tuple[float, float]:
        s = max(1, 2 * sum(isinstance(res2d, str) and "u" in res2d for res2d in self.n_res2d)) * max(
            1, 2 * len([up3d for up3d in self.n_res3d if len(up3d) == 2])
        )
        return s, s

    def get_shrinkage(self, ipt_shape: Optional[Tuple[int, int]] = None) -> Tuple[int, int]:
        s = 0
        for up3d in self.n_res3d:
            s += 2
            if len(up3d) > 1:
                s *= 2

        # s += 1  # 3d valid conv
        return s, s

    def get_output_shape(self, ipt_shape: Tuple[int, int]) -> Tuple[int, int, int]:
        return (self.z_out,) + tuple(
            i * sc - 2 * sr for i, sc, sr in zip(ipt_shape, self.get_scaling(), self.get_shrinkage())
        )


def try_static(backprop: bool = True):
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt

    from lnet.datasets import get_dataset_from_info, ZipDataset, N5CachedDataset
    from lnet.datasets.gcamp import ref0_lf, ref0_ls
    from lnet.transformations import Normalize01, ComposedTransformation, ChannelFromLightField, Cast, Crop

    m = A04(input_name="lf", prediction_name="pred", z_out=51, nnum=19)
    # n_res2d: [976, 488, u, 244, 244, u, 122, 122]
    # inplanes_3d: 7
    # n_res3d: [[7, 7], [7], [1]]

    lfds = N5CachedDataset(
        get_dataset_from_info(
            TensorInfo(
                name="lf",
                root="GHUFNAGELLFLenseLeNet_Microscope",
                location="20191031_Beads_MixedSizes/Beads_01micron_highConcentration/2019-10-31_04.57.13/stack_0_channel_0/TP_*/RC_rectified/Cam_Right_1_rectified.tif",
                insert_singleton_axes_at=[0, 0],
            )
        )
    )
    lsds = N5CachedDataset(
        get_dataset_from_info(
            TensorInfo(
                name="ls",
                root="GHUFNAGELLFLenseLeNet_Microscope",
                location="20191031_Beads_MixedSizes/Beads_01micron_highConcentration/2019-10-31_04.57.13/stack_1_channel_1/TP_*/LC/Cam_Left_registered.tif",
                insert_singleton_axes_at=[0, 0],
                transformations=[
                    {
                        "Resize": {
                            "apply_to": "ls",
                            "shape": [1.0, 121, 0.42105263157894736842105263157895, 0.42105263157894736842105263157895],
                            "order": 2,
                        }
                    }
                ],
            )
        )
    )
    trf = ComposedTransformation(
        Crop(apply_to="ls", crop=[[0, 0], [35, -35], [8, -8], [8, -8]]),
        Normalize01(apply_to=["lf", "ls"], min_percentile=0, max_percentile=100),
        ChannelFromLightField(apply_to="lf", nnum=19),
        Cast(apply_to=["lf", "ls"], dtype="float32", device="cuda"),
    )
    ds = ZipDataset({"lf": lfds, "ls": lsds}, transformation=trf)
    loader = DataLoader(ds, batch_size=1, collate_fn=get_collate_fn(lambda t: t))

    device = torch.device("cuda")
    m = m.to(device)

    # state = torch.load(checkpoint, map_location=device)
    # m.load_state_dict(state, strict=False)

    sample = next(iter(loader))
    ipt = sample["lf"]
    tgt = sample["ls"]
    # ipt = torch.rand(1, nnum ** 2, 5, 5)
    print("get_scaling", m.get_scaling(ipt.shape[2:]))
    print("get_shrinkage", m.get_shrinkage(ipt.shape[2:]))
    print("get_output_shape()", m.get_output_shape(ipt.shape[2:]))
    print("ipt", ipt.shape, "tgt", tgt.shape)
    out_sample = m(sample)
    out = out_sample["pred"]
    if backprop:
        loss_fn = torch.nn.MSELoss()
        loss = loss_fn(out, tgt)
        loss.backward()
        adam = torch.optim.Adam(m.parameters())
        adam.step()

    tgt_show = tgt[0, 0].detach().cpu().numpy()
    plt.imshow(tgt_show.max(axis=0))
    plt.title("tgt")
    plt.show()
    plt.imshow(tgt_show.max(axis=1))
    plt.title("tgt")
    plt.show()
    plt.imshow(tgt_show.max(axis=2))
    plt.title("tgt")
    plt.show()

    print("pred", out.shape)
    plt.imshow(out[0, 0].detach().cpu().numpy().max(axis=0))
    plt.title("pred")
    plt.show()
    plt.imshow(out[0, 0].detach().cpu().numpy().max(axis=1))
    plt.title("pred")
    plt.show()
    plt.imshow(out[0, 0].detach().cpu().numpy().max(axis=2))
    plt.title("pred")
    plt.show()

    print("done")


def try_dynamic():
    import matplotlib.pyplot as plt

    from torch.utils.data import DataLoader
    from lnet.datasets import get_dataset_from_info, ZipDataset
    from lnet.datasets.gcamp import ref0_lf, ref0_ls
    from lnet.datasets import N5CachedDataset
    from lnet.transformations import Normalize01

    m = A04(input_name="lf", prediction_name="pred", z_out=49, nnum=19)
    # n_res2d: [976, 488, u, 244, 244, u, 122, 122]
    # inplanes_3d: 7
    # n_res3d: [[7, 7], [7], [1]]

    # lfds = N5CachedDataset(get_dataset_from_info(ref0_lf))
    # lsds = N5CachedDataset(get_dataset_from_info(ref0_ls))
    normalize = Normalize01(apply_to=["lf", "ls"], min_percentile=0, max_percentile=100)
    ds = ZipDataset({"lf": lfds, "ls": lsds}, transformation=normalize)
    loader = DataLoader(ds, batch_size=1)
    device = torch.device("cuda")
    m = m.to(device)

    sample = next(iter(loader))
    ipt = sample["lf"]
    tgt = sample["ls"]

    z_slice = sample["meta"]["z_slice"]
    ipt, tgt = ipt.to(device), tgt.to(device)
    print("get_scaling", m.get_scaling(ipt.shape[2:]))
    print("get_shrinkage", m.get_shrinkage(ipt.shape[2:]))
    print("get_output_shape()", m.get_output_shape(ipt.shape[2:]))
    print("ipt", ipt.shape, "tgt", tgt.shape, z_slice)
    plt.imshow(tgt[0, 0].detach().cpu().numpy())
    plt.title("tgt")
    plt.show()
    # print("scale", m.get_scaling(ipt.shape[2:]))
    # print("out", m.get_output_shape(ipt.shape[2:]))
    # print("shrink", m.get_shrinkage(ipt.shape[2:]))
    # print("len 3d", len(m.res3d))

    out_sample = m(sample)
    out = out_sample["out"]
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
