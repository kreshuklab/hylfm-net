from typing import Optional

import torch.nn

import pytorch_msssim
from hylfm.hylfm_types import PeriodUnit
from hylfm.utils.general import Period


class L1(torch.nn.L1Loss):
    minimize = True

    def __call__(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        *,
        epoch: Optional[int] = None,
        iteration: Optional[int] = None,
        epoch_len: Optional[int] = None,
    ) -> torch.Tensor:
        return super().__call__(prediction, target)


class MSE(torch.nn.MSELoss):
    minimize = True

    def __call__(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        *,
        epoch: Optional[int] = None,
        iteration: Optional[int] = None,
        epoch_len: Optional[int] = None,
    ) -> torch.Tensor:
        return super().__call__(prediction, target)


class SmoothL1(torch.nn.SmoothL1Loss):
    minimize = True

    def __call__(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        *,
        epoch: Optional[int] = None,
        iteration: Optional[int] = None,
        epoch_len: Optional[int] = None,
    ) -> torch.Tensor:
        return super().__call__(prediction, target)


class SSIM(pytorch_msssim.SSIM):
    minimize = False

    def __call__(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        *,
        epoch: Optional[int] = None,
        iteration: Optional[int] = None,
        epoch_len: Optional[int] = None,
    ) -> torch.Tensor:
        return super().__call__(prediction, target)


class MS_SSIM(pytorch_msssim.MS_SSIM):
    minimize = False

    def __call__(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        *,
        epoch: Optional[int] = None,
        iteration: Optional[int] = None,
        epoch_len: Optional[int] = None,
    ) -> torch.Tensor:
        return super().__call__(prediction, target)


class WeightedLossBase(torch.nn.Module):
    def __init__(
        self,
        threshold: float,
        weight: float,
        apply_weight_above_threshold: bool,
        decay_weight_every: Period = Period(1, PeriodUnit.epoch),
        decay_weight_by: Optional[float] = None,
        decay_weight_limit: float = 1.0,
        **super_kwargs,
    ):
        super().__init__(**super_kwargs)  # noqa: init mixin
        self.threshold = threshold
        self.weight = weight
        self.apply_weight_above_threshold = apply_weight_above_threshold
        self.decay_weight_every = decay_weight_every
        self.decay_weight_by = decay_weight_by
        self.decay_weight_limit = decay_weight_limit

    def __call__(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        *,
        epoch: Optional[int] = None,
        iteration: Optional[int] = None,
        epoch_len: Optional[int] = None,
    ) -> torch.Tensor:
        pixelwise = super().__call__(prediction, target)

        weights = torch.ones_like(pixelwise)

        if self.apply_weight_above_threshold:
            mask = target >= self.threshold
        else:
            mask = target < self.threshold

        weights[mask] = self.weight
        pixelwise *= weights

        if (
            epoch is not None
            and self.decay_weight_by
            and self.decay_weight_every.match(epoch=epoch, iteration=iteration, epoch_len=epoch_len)
        ):
            self.weight = (self.weight - self.decay_weight_limit) * self.decay_weight_by + self.decay_weight_limit

        return pixelwise.mean()


class WeightedL1(WeightedLossBase, torch.nn.L1Loss):
    minimize = True

    def __init__(
        self,
        *,
        threshold: float,
        weight: float,
        apply_weight_above_threshold: bool,
        decay_weight_every: Period,
        decay_weight_by: Optional[float],
        decay_weight_limit: float,
    ):
        super().__init__(
            threshold=threshold,
            weight=weight,
            apply_weight_above_threshold=apply_weight_above_threshold,
            decay_weight_every=decay_weight_every,
            decay_weight_by=decay_weight_by,
            decay_weight_limit=decay_weight_limit,
            reduction="none",
        )


class WeightedSmoothL1(WeightedLossBase, torch.nn.SmoothL1Loss):
    minimize = True

    def __init__(
        self,
        *,
        threshold: float,
        weight: float,
        apply_weight_above_threshold: bool,
        decay_weight_every: Period,
        decay_weight_by: Optional[float],
        decay_weight_limit: float,
        beta: float = 1.0,
    ):
        super().__init__(
            threshold=threshold,
            weight=weight,
            apply_weight_above_threshold=apply_weight_above_threshold,
            decay_weight_every=decay_weight_every,
            decay_weight_by=decay_weight_by,
            decay_weight_limit=decay_weight_limit,
            beta=beta,
            reduction="none",
        )


class SmoothL1_MS_SSIM(MS_SSIM):
    minimize = True

    def __init__(self, beta: float = 1.0, ms_ssim_weight: float = 0.01, **super_kwargs):
        super().__init__(**super_kwargs)
        self.smooth_l1 = SmoothL1(beta=beta)
        self.ms_ssim_weight = ms_ssim_weight

    def __call__(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        *,
        epoch: Optional[int] = None,
        iteration: Optional[int] = None,
        epoch_len: Optional[int] = None,
    ) -> torch.Tensor:
        ms_ssim = super().__call__(prediction, target, epoch=epoch, iteration=iteration, epoch_len=epoch_len)
        smooth_l1 = self.smooth_l1(prediction, target, epoch=epoch, iteration=iteration, epoch_len=epoch_len)
        return smooth_l1 - ms_ssim * self.ms_ssim_weight


class WeightedSmoothL1_MS_SSIM(MS_SSIM):
    minimize = True

    def __init__(
        self,
        # weight kwargs
        threshold: float,
        weight: float,
        apply_weight_above_threshold: bool,
        decay_weight_every: Period,
        decay_weight_by: Optional[float],
        decay_weight_limit: float,
        # smooth l1 kwargs
        beta: float = 1.0,
        # mix ms_ssim kwargs
        ms_ssim_weight: float = 0.01,
        # ms_ssim kwargs
        **super_kwargs,
    ):
        super().__init__(**super_kwargs)
        self.weighted_smooth_l1 = WeightedSmoothL1(
            threshold=threshold,
            weight=weight,
            apply_weight_above_threshold=apply_weight_above_threshold,
            decay_weight_every=decay_weight_every,
            decay_weight_by=decay_weight_by,
            decay_weight_limit=decay_weight_limit,
            beta=beta,
        )
        self.ms_ssim_weight = ms_ssim_weight

    def __call__(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        *,
        epoch: Optional[int] = None,
        iteration: Optional[int] = None,
        epoch_len: Optional[int] = None,
    ) -> torch.Tensor:
        ms_ssim = super().__call__(prediction, target, epoch=epoch, iteration=iteration, epoch_len=epoch_len)
        weighted_smooth_l1 = self.weighted_smooth_l1(
            prediction, target, epoch=epoch, iteration=iteration, epoch_len=epoch_len
        )
        return weighted_smooth_l1 - ms_ssim * self.ms_ssim_weight
