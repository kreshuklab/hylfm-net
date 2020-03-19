from typing import Optional, Tuple

import torch
from inferno.io.transform import Transform


class MaskCenter:
    """Mask out the 'artifact plane' """

    def __init__(self, n_mask: int = 1, center: Optional[int] = None, **super_kwargs):
        super().__init__(**super_kwargs)
        assert n_mask > 0
        self.n_mask = n_mask
        self.center = center

    def __call__(self, pred: torch.Tensor, tgt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        assert len(pred.shape) == 5
        mask = torch.ones_like(pred)
        if self.center is None:
            mid_z = round(mask.shape[2] / 2)
        else:
            mid_z = self.center
        # mid_z = self.center if self.center is not None else mask.shape[2] // 2
        mid_z_start = mid_z - self.n_mask // 2
        mid_z_end = mid_z + (self.n_mask + 1) // 2
        mask[:, :, mid_z_start:mid_z_end] = 0
        mask.requires_grad = False

        # mask prediction with mask
        masked_prediction = pred * mask
        return masked_prediction, tgt


class MaskBorder(Transform):
    """Mask out the borders in x and y (aka in the last two dimensions)"""

    def __init__(self, n_mask: int = 1, **super_kwargs):
        super().__init__(**super_kwargs)
        assert n_mask > 0
        self.n_mask = n_mask

    def batch_function(self, tensors):
        prediction, target = tensors
        assert len(prediction.shape) > 3
        mask = torch.zeros_like(prediction)
        mask[..., self.n_mask : -self.n_mask, self.n_mask : -self.n_mask] = 1
        mask.requires_grad = False

        # mask prediction with mask
        masked_prediction = prediction * mask
        return masked_prediction, target
