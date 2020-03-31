from __future__ import annotations

import logging
import typing
from collections import OrderedDict

import torch.nn
from ignite.engine import Engine, Events
from torch import FloatTensor, Tensor
from torch.autograd import Variable

logger = logging.getLogger(__name__)


MSELoss = torch.nn.MSELoss


class CriterionWrapper(torch.nn.Module):
    def __init__(
        self, tensor_names: typing.Dict[str, str], criterion_class: torch.nn.Module, postfix: str = "", **kwargs
    ):
        super().__init__()
        self.input_kargs = tensor_names
        self.criterion = criterion_class(**kwargs)
        self.postfix = postfix

    def forward(self, tensors: typing.OrderedDict[str, typing.Any]):
        out = self.criterion.forward(**{name: tensors[tensor_name] for name, tensor_name in self.tensor_names.items()})
        if isinstance(out, OrderedDict):
            assert self.criterion.__class__.__name__ in out
            for loss_name, loss_value in out.items():
                loss_name += self.postfix
                assert loss_name not in tensors
                tensors[loss_name] = loss_value
        else:
            loss_name = self.criterion.__class__.__name__ + self.postfix
            assert loss_name not in tensors
            tensors[loss_name] = out

        return tensors


class WeightedL1Loss(torch.nn.L1Loss):
    def __init__(
        self,
        engine: Engine,
        threshold: float,
        initial_weight: float,
        decay_by: float,
        every_nth_epoch: int,
        apply_below_threshold: bool = False,
    ):
        super().__init__(reduction="none")
        self.threshold = threshold
        self.apply_below_threshold = apply_below_threshold

        self.weight = initial_weight - 1.0

        @engine.on(Events.EPOCH_COMPLETED)
        def decay_weight(engine):
            if engine.state.epoch % every_nth_epoch == 0:
                self.weight *= decay_by
                logger.info("decayed loss weight to %f (+1.0)", self.weight)

    def forward(self, input, target):
        l1 = super().forward(input, target)

        if self.apply_below_threshold:
            mask = target < self.threshold
        else:
            mask = target >= self.threshold

        if self.training:
            l1_additional_weights = torch.zeros_like(l1)
            l1_additional_weights[mask] = l1[mask] * self.weight
            l1 = l1 + l1_additional_weights

        return OrderedDict([(self.__class__.__name__, l1.mean()), (f"{self.__class__.__name__} pixelwise", l1)])


class WeightedSmoothL1Loss(torch.nn.SmoothL1Loss):
    def __init__(
        self,
        engine: Engine,
        threshold: float,
        initial_weight: float,
        decay_by: float,
        every_nth_epoch: int,
        apply_below_threshold: bool = False,
    ):
        super().__init__(reduction="none")
        self.threshold = threshold
        self.apply_below_threshold = apply_below_threshold

        self.weight = initial_weight - 1.0

        @engine.on(Events.EPOCH_COMPLETED)
        def decay_weight(engine):
            if engine.state.epoch % every_nth_epoch == 0:
                self.weight *= decay_by
                logger.info("decayed loss weight to %f (+1.0)", self.weight)

    def forward(self, input, target):
        l1 = super().forward(input, target)

        if self.training:
            if self.apply_below_threshold:
                mask = target < self.threshold
            else:
                mask = target >= self.threshold

            l1_additional_weights = torch.zeros_like(l1)
            l1_additional_weights[mask] = l1[mask] * self.weight
            l1 = l1 + l1_additional_weights

        return OrderedDict([(self.__class__.__name__, l1.mean()), (f"{self.__class__.__name__} pixelwise", l1)])


class BCEWithLogitsLoss(torch.nn.BCEWithLogitsLoss):
    def __init__(self, weight=None, pos_weight=None, **kwargs):
        if weight is not None:
            weight = FloatTensor(weight)

        if pos_weight is not None:
            pos_weight = FloatTensor(pos_weight)

        super().__init__(weight=weight, pos_weight=pos_weight, **kwargs)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        raise NotImplementedError


def flatten_samples(tensor_or_variable):
    """
    Flattens a tensor or a variable such that the channel axis is first and the sample axis
    is second. The shapes are transformed as follows:
        (N, C, H, W) --> (C, N * H * W)
        (N, C, D, H, W) --> (C, N * D * H * W)
        (N, C) --> (C, N)
    The input must be atleast 2d.
    """
    assert (
        tensor_or_variable.dim() >= 2
    ), f"Tensor or variable must be atleast 2D. Got one of dim {tensor_or_variable.dim()}."
    # Get number of channels
    num_channels = tensor_or_variable.size(1)
    # Permute the channel axis to first
    permute_axes = list(range(tensor_or_variable.dim()))
    permute_axes[0], permute_axes[1] = permute_axes[1], permute_axes[0]
    # For input shape (say) NCHW, this should have the shape CNHW
    permuted = tensor_or_variable.permute(*permute_axes).contiguous()
    # Now flatten out all but the first axis and return
    flattened = permuted.view(num_channels, -1)
    return flattened


class SorensenDiceLoss(torch.nn.Module):
    """
    adapted from inferno
    Computes a loss scalar, which when minimized maximizes the Sorensen-Dice similarity
    between the input and the target. For both inputs and targets it must be the case that
    `input_or_target.size(1) = num_channels`.
    """

    def __init__(self, weight=None, channelwise=True, eps=1e-6):
        """
        Parameters
        ----------
        weight : torch.FloatTensor or torch.cuda.FloatTensor
            Class weights. Applies only if `channelwise = True`.
        channelwise : bool
            Whether to apply the loss channelwise and sum the results (True)
            or to apply it on all channels jointly (False).
        """
        super(SorensenDiceLoss, self).__init__()
        self.register_buffer("weight", weight)
        self.channelwise = channelwise
        self.eps = eps

    def forward(self, input: torch.FloatTensor, target: torch.FloatTensor):
        """
        input:      torch.FloatTensor or torch.cuda.FloatTensor
        target:     torch.FloatTensor or torch.cuda.FloatTensor

        Expected shape of the inputs: (batch_size, nb_channels, ...)
        """
        assert input.size() == target.size()
        if not self.channelwise:
            numerator = (input * target).sum()
            denominator = (input * input).sum() + (target * target).sum()
            loss = -2.0 * (numerator / denominator.clamp(min=self.eps))
        else:
            # Flatten input and target to have the shape (C, N),
            # where N is the number of samples
            input = flatten_samples(input)
            target = flatten_samples(target)
            # Compute numerator and denominator (by summing over samples and
            # leaving the channels intact)
            numerator = (input * target).sum(-1)
            denominator = (input * input).sum(-1) + (target * target).sum(-1)
            channelwise_loss = -2 * (numerator / denominator.clamp(min=self.eps))
            if self.weight is not None:
                # With pytorch < 0.2, channelwise_loss.size = (C, 1).
                if channelwise_loss.dim() == 2:
                    channelwise_loss = channelwise_loss.squeeze(1)
                # Wrap weights in a variable
                weight = Variable(self.weight, requires_grad=False)
                assert weight.size() == channelwise_loss.size()
                # Apply weight
                channelwise_loss = weight * channelwise_loss
            # Sum over the channels to compute the total loss
            loss = channelwise_loss.sum()
        return loss
