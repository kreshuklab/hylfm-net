from torch import nn


class Transition2D(nn.Sequential):
    def __init__(self, num_input_features, num_output_features, batch_norm: bool = True):
        super().__init__()
        if batch_norm:
            self.add_module("norm", nn.BatchNorm2d(num_input_features))

        self.add_module("relu", nn.ReLU(inplace=True))
        self.add_module(
            "conv2d", nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False)
        )


class Transition3D(nn.Sequential):
    def __init__(self, num_input_features, num_output_features, batch_norm: bool = True):
        super().__init__()
        if batch_norm:
            self.add_module("norm", nn.BatchNorm3d(num_input_features))

        self.add_module("relu", nn.ReLU(inplace=True))
        self.add_module(
            "conv3d", nn.Conv3d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False)
        )
