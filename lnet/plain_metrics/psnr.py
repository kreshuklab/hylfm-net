import torch
import torch.nn.functional
from ignite.exceptions import NotComputableError
from skimage.measure.simple_metrics import compare_psnr

from .base import Metric


def psnr_skimage(pred, target, **kwargs):
    assert pred.shape == target.shape
    # average psnr over batch dim
    ret = 0
    for p, t in zip(pred, target):
        ret += compare_psnr(im_test=p, im_true=t, **kwargs)

    return ret / target.shape[0]


class PSNR_SkImage(Metric):
    def __init__(self, *, data_range=None, **super_kwargs):
        super().__init__(**super_kwargs)
        self.data_range = data_range

    def reset(self):
        self._sum = 0.0
        self._num_examples = 0

    def update_impl(self, *, pred, tgt):
        n = pred.shape[0]
        self._sum += (
            psnr_skimage(
                pred.flatten(end_dim=1).numpy(), tgt.flatten(end_dim=1).numpy(), data_range=self.data_range
            )  # bc > b
            * n
        )
        self._num_examples += n

    def compute_impl(self):
        if self._num_examples == 0:
            raise NotComputableError("PSNR_SKImage must have at least one example before it can be computed.")

        return self._sum / self._num_examples


def psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float) -> float:
    assert pred.shape == target.shape
    if data_range is None:
        data_range = target.max() - target.min()

    ret = 0
    with torch.no_grad():
        # average psnr over batch dim
        for p, t in zip(pred, target):
            ret += 10 * torch.log10(data_range ** 2 / torch.nn.functional.mse_loss(p, t)).item()

    return ret / target.shape[0]


class PSNR(Metric):
    def __init__(self, *, data_range=None, pred: str = "pred", tgt: str = "tgt", **super_kwargs):
        super().__init__(pred=pred, tgt=tgt, **super_kwargs)
        self.data_range = data_range

    def reset(self):
        self._sum = 0.0
        self._num_examples = 0

    def update_impl(self, *, pred, tgt):
        n = pred.shape[0]
        self._sum += psnr(pred.flatten(end_dim=1), tgt.flatten(end_dim=1), data_range=self.data_range) * n  # bc > b
        self._num_examples += n

    def compute_impl(self):
        if self._num_examples == 0:
            raise NotComputableError("PSNR must have at least one example before it can be computed.")

        return self._sum / self._num_examples
