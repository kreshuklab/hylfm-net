import logging
import sys
import matplotlib.pyplot as plt
import numpy
import torch
import torch.nn as nn
from torchvision.models.resnet import BasicBlock, Bottleneck

from functools import partial
from inferno.extensions.layers.convolutional import (
    ConvActivation,
    ValidConvActivation,
    ConvReLU2D,
    ValidConvReLU2D,
    ConvReLU3D,
    ValidConvReLU3D,
    BNReLUConv2D,
    ValidBNReLUConv2D,
    BNReLUConv3D,
    ValidBNReLUConv3D,
)
from inferno.extensions.initializers import (
    OrthogonalWeightsZeroBias,
    KaimingNormalWeightsZeroBias,
    Initialization,
    Constant,
)
from inferno.extensions.layers.reshape import Flatten

from .structural_layers import Crop

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


class Conv2D(ConvActivation):
    """
    2D convolutional layer with same padding and Kaiming-He initialization.
    By default, this layer does not apply an activation function.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dilation=1,
        stride=1,
        activation=None,
        initialization=KaimingNormalWeightsZeroBias,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            stride=stride,
            dim=2,
            activation=activation,
            initialization=initialization(),
        )


class ValidConv2D(ValidConvActivation):
    """
    2D convolutional layer with valid padding and Kaiming-He initialization.
    By default, this layer does not apply an activation function.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dilation=1,
        stride=1,
        activation=None,
        initialization=KaimingNormalWeightsZeroBias,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            stride=stride,
            dim=2,
            activation=activation,
            initialization=initialization(),
        )


class Conv3D(ConvActivation):
    """
    2D convolutional layer with same padding and Kaiming-He initialization.
    By default, this layer does not apply an activation function.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dilation=1,
        stride=1,
        activation=None,
        initialization=KaimingNormalWeightsZeroBias,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            stride=stride,
            dim=3,
            activation=activation,
            initialization=initialization(),
        )


class ValidConv3D(ValidConvActivation):
    """
    2D convolutional layer with valid padding and Kaiming-He initialization.
    By default, this layer does not apply an activation function.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dilation=1,
        stride=1,
        activation=None,
        initialization=KaimingNormalWeightsZeroBias,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            stride=stride,
            dim=3,
            activation=activation,
            initialization=initialization(),
        )


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
        self, in_n_filters, n_filters, kernel_size=(3, 3), batch_norm=False, conv_per_block=2, valid: bool = False
    ):
        super().__init__()
        assert conv_per_block >= 2
        self.debug = False  #  sys.gettrace() is not None
        logger.debug(
            "%dD Resnet Block with n_filters=%d, kernel_size=%s, valid=%r",
            len(kernel_size),
            n_filters,
            kernel_size,
            valid,
        )

        if len(kernel_size) == 2:
            if batch_norm and valid:
                BNConvReLU = ValidBNReLUConv2D
            elif batch_norm:
                BNConvReLU = BNReLUConv2D
            elif valid:
                BNConvReLU = ValidConvReLU2D
            else:
                BNConvReLU = ConvReLU2D

            # ConvReLU = ConvReLU2D
            if valid:
                Conv = ValidConv2D
            else:
                Conv = Conv2D
        elif len(kernel_size) == 3:
            if batch_norm and valid:
                BNConvReLU = ValidBNReLUConv3D
            elif batch_norm:
                BNConvReLU = BNReLUConv3D
            elif valid:
                BNConvReLU = ValidConvReLU3D
            else:
                BNConvReLU = ConvReLU3D
            # ConvReLU = ConvReLU3D
            if valid:
                Conv = ValidConv3D
            else:
                Conv = Conv3D
        else:
            raise ValueError(kernel_size)

        layers = []
        layers.append(BNConvReLU(in_channels=in_n_filters, out_channels=n_filters, kernel_size=kernel_size))

        for _ in range(conv_per_block - 2):
            layers.append(BNConvReLU(n_filters, n_filters, kernel_size))

        layers.append(Conv(n_filters, n_filters, kernel_size))
        if batch_norm:
            raise NotImplementedError

        self.block = nn.Sequential(*layers)
        # self.block = DebugSequential(*layers)

        if n_filters != in_n_filters:
            logger.debug("Resnet Block: Add projection layer to input")
            self.projection_layer = Conv(in_n_filters, n_filters, kernel_size=1)
        else:
            self.projection_layer = None

        if valid:
            crop_ech_side = [conv_per_block * (ks // 2) for ks in kernel_size]
            self.crop = Crop(..., *[slice(c, -c) for c in crop_ech_side])
        else:
            self.crop = None

        self.relu = nn.ReLU()

        # determine shrinkage
        self.shrinkage = (1, 1) + tuple([conv_per_block * (ks - 1) for ks in kernel_size])

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
