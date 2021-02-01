import torch.nn

import pytorch_msssim


class L1Loss(torch.nn.L1Loss):
    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return super().forward(input=prediction, target=target)


class MSELoss(torch.nn.MSELoss):
    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return super().forward(input=prediction, target=target)


class SmoothL1Loss(torch.nn.SmoothL1Loss):
    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return super().forward(input=prediction, target=target)


class SSIM(pytorch_msssim.SSIM):
    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return super().forward(X=prediction, Y=target)


class MS_SSIM(pytorch_msssim.MS_SSIM):
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


class WeightedL1Loss(WeightedLossBase, torch.nn.L1Loss):
    def __init__(self, *, threshold: float, weight: float, apply_below_threshold: bool):
        super().__init__(
            threshold=threshold, weight=weight, apply_below_threshold=apply_below_threshold, reduction="none"
        )


class WeightedSmoothL1Loss(WeightedLossBase, torch.nn.SmoothL1Loss):
    def __init__(self, *, threshold: float, weight: float, apply_below_threshold: bool, beta: float = 1.0):
        super().__init__(
            threshold=threshold, weight=weight, apply_below_threshold=apply_below_threshold, beta=beta, reduction="none"
        )
