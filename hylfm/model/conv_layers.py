import logging

import matplotlib.pyplot as plt
import numpy
import torch.nn as nn

from inferno.extensions.initializers import KaimingNormalWeightsZeroBias
from inferno.extensions.layers import convolutional as inferno_convolutional

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


from .structural_layers import Crop


Conv2D = inferno_convolutional.Conv2D
Conv3D = inferno_convolutional.Conv3D
ValidConv2D = inferno_convolutional.ValidConv2D
ValidConv3D = inferno_convolutional.ValidConv3D


logger = logging.getLogger(__name__)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    ret = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)
    ret.apply(KaimingNormalWeightsZeroBias())
    return ret


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    ret = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=True)
    ret.apply(KaimingNormalWeightsZeroBias())
    return ret


def conv3d3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    ret = nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)
    ret.apply(KaimingNormalWeightsZeroBias())
    return ret


def conv3d1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    ret = nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=True)
    ret.apply(KaimingNormalWeightsZeroBias())
    return ret


class BasicBlock3dNoBatchNorm(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3d3x3(inplanes, planes, stride)
        # self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3d3x3(planes, planes)

        # self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        # out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        # out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class BasicBlockNoBatchNorm(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        # self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)

        # self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        # out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        # out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class BottleneckNoBatchNorm(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv1x1(inplanes, planes)
        # self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        # self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        # self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        # out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        # out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        # out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class DebugSequential(nn.Sequential):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        for child in self:
            x = child(x)
            # import matplotlib.pyplot as plt
            # plt.title(str(child.__class__) + " " + str(x.shape))
            # if len(x.shape) == 4:
            #     plt.imshow(x[0].detach().cpu().numpy().max(axis=0))
            # elif len(x.shape) == 5:
            #     plt.imshow(x[0].detach().cpu().numpy().max(axis=0).max(axis=0))
            # else:
            #     assert False
            #
            # plt.colorbar()
            # plt.show()

        return x


class ResnetBlock(nn.Module):
    def __init__(
        self,
        in_n_filters,
        n_filters,
        kernel_size=(3, 3),
        batch_norm=False,
        conv_per_block=2,
        valid: bool = False,
        activation: Literal["ReLU", "ELU", "Sigmoid", "SELU", ""] = "ReLU",
    ):
        super().__init__()
        if batch_norm and activation != "ReLU":
            raise NotImplementedError("batch_norm with non ReLU activation")

        assert isinstance(kernel_size, tuple), kernel_size
        assert conv_per_block >= 2
        self.debug = False  #  sys.gettrace() is not None
        logger.debug(
            "%dD Resnet Block with n_filters=%d, kernel_size=%s, valid=%r",
            len(kernel_size),
            n_filters,
            kernel_size,
            valid,
        )

        Conv = getattr(
            inferno_convolutional,
            f"{'BNReLU' if batch_norm else ''}{'Valid' if valid else ''}Conv{'' if batch_norm else activation}{len(kernel_size)}D",
        )
        FinalConv = getattr(
            inferno_convolutional, f"{'BNReLU' if batch_norm else ''}{'Valid' if valid else ''}Conv{len(kernel_size)}D"
        )

        layers = []
        layers.append(Conv(in_channels=in_n_filters, out_channels=n_filters, kernel_size=kernel_size))

        for _ in range(conv_per_block - 2):
            layers.append(Conv(n_filters, n_filters, kernel_size))

        layers.append(FinalConv(n_filters, n_filters, kernel_size))

        self.block = nn.Sequential(*layers)

        if n_filters != in_n_filters:
            logger.debug("Resnet Block: Add projection layer to input")
            ProjConv = getattr(inferno_convolutional, f"Conv{len(kernel_size)}D")
            self.projection_layer = ProjConv(in_n_filters, n_filters, kernel_size=1)
        else:
            self.projection_layer = None

        if valid:
            crop_each_side = [conv_per_block * (ks // 2) for ks in kernel_size]
            self.crop = Crop(..., *[slice(c, -c) for c in crop_each_side])
        else:
            self.crop = None

        self.relu = nn.ReLU()

        # determine shrinkage
        # self.shrinkage = (1, 1) + tuple([conv_per_block * (ks - 1) for ks in kernel_size])

    def forward(self, input):
        x = self.block(input)
        if self.debug:
            plt.title(f"after block {x.shape}")
            if len(x.shape) == 4:
                plt.imshow(x[0].detach().cpu().numpy().max(axis=0).astype(numpy.float32))
            elif len(x.shape) == 5:
                plt.imshow(x[0].detach().cpu().numpy().max(axis=0).max(axis=0).astype(numpy.float32))
            else:
                assert False

            plt.colorbar()
            plt.show()

        if self.crop is not None:
            input = self.crop(input)

        if self.projection_layer is None:
            x = x + input
        else:
            projected = self.projection_layer(input)
            x = x + projected
            if self.debug:
                plt.title(f"projection {projected.shape}")
                if len(projected.shape) == 4:
                    plt.imshow(projected[0].detach().cpu().numpy().max(axis=0).astype(numpy.float32))
                elif len(projected.shape) == 5:
                    plt.imshow(projected[0].detach().cpu().numpy().max(axis=0).max(axis=0).astype(numpy.float32))
                else:
                    assert False

                plt.colorbar()
                plt.show()

        if self.debug:
            plt.title(f"Add {x.shape}")
            if len(x.shape) == 4:
                plt.imshow(x[0].detach().cpu().numpy().max(axis=0).astype(numpy.float32))
            elif len(x.shape) == 5:
                plt.imshow(x[0].detach().cpu().numpy().max(axis=0).max(axis=0).astype(numpy.float32))
            else:
                assert False

            plt.colorbar()
            plt.show()

        x = self.relu(x)
        if self.debug:
            plt.title(f"ReLU {x.shape}")
            if len(x.shape) == 4:
                plt.imshow(x[0].detach().cpu().numpy().max(axis=0).astype(numpy.float32))
            elif len(x.shape) == 5:
                plt.imshow(x[0].detach().cpu().numpy().max(axis=0).max(axis=0).astype(numpy.float32))
            else:
                assert False

            plt.colorbar()
            plt.show()

        return x
