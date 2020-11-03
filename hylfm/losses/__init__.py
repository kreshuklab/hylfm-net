import torch.nn

import pytorch_msssim
from hylfm.losses import dice
from hylfm.losses.on_tensors import LossOnTensorsTorchMixin
from hylfm.losses.weighted import WeightedLossOnTensorsTorchMixin


class L1Loss(LossOnTensorsTorchMixin, torch.nn.L1Loss):
    pass


class MSELoss(LossOnTensorsTorchMixin, torch.nn.MSELoss):
    pass


class SmoothL1Loss(LossOnTensorsTorchMixin, torch.nn.SmoothL1Loss):
    pass


class SSIM(LossOnTensorsTorchMixin, pytorch_msssim.SSIM):
    is_inverted_metric = True


class MS_SSIM(LossOnTensorsTorchMixin, pytorch_msssim.MS_SSIM):
    is_inverted_metric = True


class WeightedL1Loss(WeightedLossOnTensorsTorchMixin, torch.nn.L1Loss):
    pass


class WeightedSmoothL1Loss(WeightedLossOnTensorsTorchMixin, torch.nn.SmoothL1Loss):
    pass


# todo: check implementation
# class SorensenDiceLoss(LossOnTensorsTorchMixin, dice.SorensenDiceLoss):
#     pass
