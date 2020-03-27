from typing import Callable

import ignite
import numpy
import torch
import torch.nn.functional
from ignite.exceptions import NotComputableError

from skimage.measure.simple_metrics import compare_nrmse


def nrmse_skimage(pred, target, **kwargs):
    assert pred.shape == target.shape
    return compare_nrmse(target, pred, **kwargs)


class NRMSE_SkImage(ignite.metrics.Metric):
    def __init__(self, *, norm_type="Euclidean", **super_kwargs):
        super().__init__(**super_kwargs)
        self.norm_type = norm_type

    def reset(self):
        self._y_pred = []
        self._y = []

    def update(self, output):
        y_pred, y = output
        self._y_pred.append(y_pred)
        self._y.append(y)

    def compute(self):
        if not self._y:
            raise NotComputableError("NRMSE must have at least one example before it can be computed.")

        y_pred = torch.stack(self._y_pred).numpy()
        y = torch.stack(self._y).numpy()
        return nrmse_skimage(y_pred, y, norm_type=self.norm_type)

    forward = staticmethod(nrmse_skimage)


# def nrmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
#     assert pred.shape == target.shape
#     with torch.no_grad():
#         return torch.sqrt(torch.nn.functional.mse_loss(pred, target)) / torch.sqrt(torch.mean(target ** 2)).item()
#         # return torch.sqrt(torch.sum((pred - target) ** 2) / torch.sum(target ** 2))


class NRMSE(ignite.metrics.Metric):
    def reset(self):
        self._num_samples: int = 0
        self._mse: float = 0.0
        self._norm: float = 0.0

    def update(self, output):
        y_pred, y = output
        n = y.shape[0]
        self._num_samples += n
        self._mse += torch.nn.functional.mse_loss(y_pred, y).item() * n
        self._norm += torch.mean(y ** 2).item() * n

    def compute(self):
        if not self._y:
            raise NotComputableError("NRMSE must have at least one example before it can be computed.")

        return numpy.sqrt(self._mse / self._num_samples) / numpy.sqrt(self._norm / self._num_samples)
