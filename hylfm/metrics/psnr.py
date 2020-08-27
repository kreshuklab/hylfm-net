from math import log10
from typing import Dict

import torch
import torch.nn.functional
from ignite.exceptions import NotComputableError
from skimage.measure.simple_metrics import compare_psnr

from .base import Metric


class PSNR_SkImage(Metric):
    def __init__(self, *, data_range=None, **super_kwargs):
        super().__init__(**super_kwargs)
        self.data_range = data_range

    def reset(self):
        self._sum = 0.0
        self._num_examples = 0

    def update_impl(self, *, pred, tgt):
        n = pred.shape[0]
        self._sum += sum(
            compare_psnr(im_test=p, im_true=t, data_range=self.data_range)
            for p, t in zip(pred.cpu().numpy(), tgt.cpu().numpy())
        )
        self._num_examples += n

    def compute_impl(self):
        if self._num_examples == 0:
            raise NotComputableError("PSNR_SKImage must have at least one example before it can be computed.")

        return self._sum / self._num_examples


class PSNR(Metric):
    def __init__(self, *super_args, tensor_names: Dict[str, str], data_range: float, **super_kwargs):
        if "pred" not in tensor_names:
            tensor_names["pred"] = "pred"

        if "tgt" not in tensor_names:
            tensor_names["tgt"] = "tgt"

        assert len(tensor_names) == 2

        super().__init__(*super_args, tensor_names=tensor_names, **super_kwargs)
        self.log10dr20 = 20 * log10(data_range)

    def reset(self):
        self._sum = 0.0
        self._num_examples = 0

    def update_impl(self, *, pred, tgt):
        n = pred.shape[0]
        self._sum += sum(
            self.log10dr20 - 10 * torch.log10(torch.nn.functional.mse_loss(p, t)).item() for p, t in zip(pred, tgt)
        )
        self._num_examples += n

    def compute_impl(self):
        if self._num_examples == 0:
            raise NotComputableError("PSNR must have at least one example before it can be computed.")

        return self._sum / self._num_examples
