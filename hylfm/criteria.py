import torch.nn

import pytorch_msssim


class L1(torch.nn.L1Loss):
    minimize = True

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return super().forward(input=prediction, target=target)


class MSE(torch.nn.MSELoss):
    minimize = True

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return super().forward(input=prediction, target=target)


class SmoothL1(torch.nn.SmoothL1Loss):
    minimize = True

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return super().forward(input=prediction, target=target)


class SSIM(pytorch_msssim.SSIM):
    minimize = False

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return super().forward(X=prediction, Y=target)


class MS_SSIM(pytorch_msssim.MS_SSIM):
    minimize = False

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return super().forward(X=prediction, Y=target)


class WeightedLossBase(torch.nn.Module):
    def __init__(self, threshold: float, weight: float, apply_below_threshold: bool, **super_kwargs):
        super().__init__(**super_kwargs)  # noqa: init mixin
        self.threshold = threshold
        self.weight = weight
        self.apply_below_threshold = apply_below_threshold

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pixelwise = super().forward(input=prediction, target=target)

        weights = torch.ones_like(pixelwise)

        if self.apply_below_threshold:
            mask = target < self.threshold
        else:
            mask = target >= self.threshold

        weights[mask] = self.weight
        pixelwise *= weights
        return pixelwise.mean()


class WeightedL1(WeightedLossBase, torch.nn.L1Loss):
    minimize = True

    def __init__(self, *, threshold: float, weight: float, apply_below_threshold: bool):
        super().__init__(
            threshold=threshold, weight=weight, apply_below_threshold=apply_below_threshold, reduction="none"
        )


class WeightedSmoothL1(WeightedLossBase, torch.nn.SmoothL1Loss):
    minimize = True

    def __init__(self, *, threshold: float, weight: float, apply_below_threshold: bool, beta: float = 1.0):
        super().__init__(
            threshold=threshold, weight=weight, apply_below_threshold=apply_below_threshold, beta=beta, reduction="none"
        )


class SmoothL1_MS_SSIM(MS_SSIM):
    minimize = True

    def __init__(self, beta: float = 1.0, ms_ssim_weight: float = 0.01, **super_kwargs):
        super().__init__(**super_kwargs)
        self.smooth_l1 = SmoothL1(beta=beta)
        self.ms_ssim_weight = ms_ssim_weight

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ms_ssim = super().forward(prediction=prediction, target=target)
        smooth_l1 = self.smooth_l1(prediction=prediction, target=target)
        return smooth_l1 - ms_ssim * self.ms_ssim_weight
