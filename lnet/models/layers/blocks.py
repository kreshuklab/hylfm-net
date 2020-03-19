import torch.nn
from torchvision.models.resnet import BasicBlock, Bottleneck, conv1x1


class ResNet(torch.nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.relu = torch.nn.ReLU(inplace=True)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    torch.nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    torch.nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = torch.nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride), torch.nn.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


resnet_block2 = BasicBlock()


def old_resnet_block2(
    n_filters,
    kernel_size=(3, 3),
    batch_norm=False,
    pool=(1, 1),
    conv_per_block=2,
    kernel_initializer="he_normal",
    activation="relu",
):
    assert conv_per_block >= 2
    print(f"Resnet Block with n_filters = {n_filters}, pool  = {pool}, kernel_size = {kernel_size} ")

    def f(inp):
        x = Conv2D(
            n_filters,
            kernel_size,
            padding="same",
            use_bias=(not batch_norm),
            kernel_initializer=kernel_initializer,
            strides=pool,
        )(inp)
        if batch_norm:
            x = BatchNormalization(axis=-1)(x)
        x = Activation(activation)(x)

        for _ in range(conv_per_block - 2):
            x = Conv2D(
                n_filters, kernel_size, padding="same", use_bias=(not batch_norm), kernel_initializer=kernel_initializer
            )(x)
            if batch_norm:
                x = BatchNormalization(axis=-1)(x)
            x = Activation(activation)(x)

        x = Conv2D(
            n_filters, kernel_size, padding="same", use_bias=(not batch_norm), kernel_initializer=kernel_initializer
        )(x)
        if batch_norm:
            x = BatchNormalization(axis=-1)(x)

        if any(p != 1 for p in pool) or n_filters != K.int_shape(inp)[-1]:
            print("Resnet Block: Add projection layer to input")
            inp = Conv2D(
                n_filters,
                (1, 1),
                padding="same",
                use_bias=(not batch_norm),
                kernel_initializer=kernel_initializer,
                strides=pool,
            )(inp)

        x = Add()([inp, x])
        x = Activation(activation)(x)
        return x

    return f


def old_resnet_block3(
    n_filters,
    kernel_size=(3, 3, 3),
    batch_norm=False,
    pool=(1, 1, 1),
    conv_per_block=2,
    kernel_initializer="he_normal",
    activation="relu",
):
    assert conv_per_block >= 2
    print(f"Resnet Block with n_filters = {n_filters}, pool  = {pool}, kernel_size = {kernel_size} ")

    def f(inp):
        x = Conv3D(
            n_filters,
            kernel_size,
            padding="same",
            use_bias=(not batch_norm),
            kernel_initializer=kernel_initializer,
            strides=pool,
        )(inp)
        if batch_norm:
            x = BatchNormalization(axis=-1)(x)
        x = Activation(activation)(x)

        for _ in range(conv_per_block - 2):
            x = Conv3D(
                n_filters, kernel_size, padding="same", use_bias=(not batch_norm), kernel_initializer=kernel_initializer
            )(x)
            if batch_norm:
                x = BatchNormalization(axis=-1)(x)
            x = Activation(activation)(x)

        x = Conv3D(
            n_filters, kernel_size, padding="same", use_bias=(not batch_norm), kernel_initializer=kernel_initializer
        )(x)
        if batch_norm:
            x = BatchNormalization(axis=-1)(x)

        if any(p != 1 for p in pool) or n_filters != K.int_shape(inp)[-1]:
            print("Resnet Block: Add projection layer to input")
            inp = Conv3D(
                n_filters,
                (1, 1, 1),
                padding="same",
                use_bias=(not batch_norm),
                kernel_initializer=kernel_initializer,
                strides=pool,
            )(inp)

        x = Add()([inp, x])
        x = Activation(activation)(x)
        return x

    return f
