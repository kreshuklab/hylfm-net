from typing import Dict, Sequence, Union

import numpy
import torch
import torch.nn.functional
from ignite.exceptions import NotComputableError
from skimage.measure.simple_metrics import compare_nrmse

from .scale_minimize_vs import ScaleMinimizeVsMetric


def nrmse_skimage(pred, target, **kwargs):
    assert pred.shape == target.shape
    return compare_nrmse(target, pred, **kwargs)


class NRMSE_SkImage(ScaleMinimizeVsMetric):
    def __init__(self, *, norm_type="Euclidean", pred: str = "pred", tgt: str = "tgt", **super_kwargs):
        super().__init__(pred=pred, tgt=tgt, **super_kwargs)
        self.norm_type = norm_type

    def reset(self):
        self._preds = []
        self._tgts = []

    def update_impl(self, *, pred, tgt):
        self._preds.append(pred)
        self._tgts.append(tgt)

    def compute_impl(self):
        if not self._preds:
            raise NotComputableError("NRMSE must have at least one example before it can be computed.")

        pred = torch.stack(self._preds).numpy()
        tgt = torch.stack(self._tgts).numpy()
        return nrmse_skimage(pred, tgt, norm_type=self.norm_type)

    # forward = staticmethod(nrmse_skimage)


# def nrmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
#     assert pred.shape == target.shape
#     with torch.no_grad():
#         return torch.sqrt(torch.nn.functional.mse_loss(pred, target)) / torch.sqrt(torch.mean(target ** 2)).item()
#         # return torch.sqrt(torch.sum((pred - target) ** 2) / torch.sum(target ** 2))


class NRMSE(ScaleMinimizeVsMetric):
    higher_is_better = False

    def __init__(self, *super_args, tensor_names: Union[Sequence[str], Dict[str, str]], **super_kwargs):
        assert len(tensor_names) == 2
        super().__init__(*super_args, tensor_names=tensor_names, **super_kwargs)

    def reset(self):
        self._num_samples: int = 0
        self._mse: float = 0.0
        self._norm: float = 0.0

    def update_impl(self, pred, tgt):
        n = pred.shape[0]
        self._num_samples += n
        self._mse += torch.nn.functional.mse_loss(pred, tgt).item() * n
        self._norm += torch.mean(tgt ** 2).item() * n

    def compute_impl(self):
        if self._num_samples == 0:
            raise NotComputableError("NRMSE must have at least one example before it can be computed.")

        return float(numpy.sqrt(self._mse / self._num_samples) / numpy.sqrt(self._norm / self._num_samples))
