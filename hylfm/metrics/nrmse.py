from typing import Any, Dict, List, Optional, Sequence, Union

import numpy
import torch
import torch.nn.functional
from skimage.measure.simple_metrics import compare_nrmse
from torch import no_grad

from hylfm.hylfm_types import Array
from hylfm.metrics import Metric
from hylfm.metrics.base import SimpleSingleValueMetric

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


def nrmse_skimage(pred, target, **kwargs):
    assert pred.shape == target.shape
    return compare_nrmse(target, pred, **kwargs)


class NRMSE_SkImage(Metric):
    """use only for testing torch NRMSE; saves everything to RAM!"""

    preds: list
    tgts: list

    def __init__(self, *, norm_type="Euclidean", **super_kwargs):
        super().__init__(along_dim=None, per_sample=False, **super_kwargs)
        self.norm_type = norm_type

    def reset(self):
        self.preds = []
        self.tgts = []

    def __call__(self, prediction, target):
        self.preds.append(prediction)
        self.tgts.append(target)

    def compute(self):
        if not self.preds:
            raise RuntimeError("NRMSE must have at least one example before it can be computed.")

        pred = torch.stack(self.preds).numpy()
        tgt = torch.stack(self.tgts).numpy()
        return {self.name: nrmse_skimage(pred, tgt, norm_type=self.norm_type)}


class NRMSE(SimpleSingleValueMetric):
    """
    RMSE normalized by eucledian norm of target
    """

    @torch.no_grad()
    def __call__(self, prediction, target):
        prediction = torch.from_numpy(prediction) if isinstance(prediction, numpy.ndarray) else prediction
        target = torch.from_numpy(target) if isinstance(target, numpy.ndarray) else target
        return torch.sqrt(torch.nn.functional.mse_loss(prediction, target) / torch.mean(target ** 2)).item()
