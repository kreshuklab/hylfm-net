import logging
from typing import Dict, Callable

import ignite.metrics
import numpy
from ignite.exceptions import NotComputableError
from pytorch_msssim import msssim, ssim
from skimage.measure import compare_ssim

logger = logging.getLogger(__name__)


class MSSSIM(ignite.metrics.Metric):
    def __init__(self, *, window_size=11, size_average=True, val_range=None, normalize=False, **super_kwargs):
        super().__init__(**super_kwargs)
        self.normalize = normalize
        self.size_average = size_average
        self.val_range = val_range
        self.window_size = window_size

    def reset(self):
        self._sum = 0.0
        self._num_examples = 0

    def update(self, output):
        pred, tgt = output
        n = tgt.shape[0]
        pred_z_as_batch = pred.transpose(1, 2).flatten(end_dim=-4) if len(pred.shape) == 5 else pred
        tgt_z_as_batch = tgt.transpose(1, 2).flatten(end_dim=-4) if len(tgt.shape) == 5 else tgt
        value = (
            msssim(
                pred_z_as_batch,
                tgt_z_as_batch,
                normalize=self.normalize,
                size_average=self.size_average,
                val_range=self.val_range,
                window_size=self.window_size,
            ).item()
            * n
        )
        if numpy.isfinite(value):
            self._sum += value
        else:
            logger.warning("Encountered %s in MS-SSIM", value)

        self._num_examples += n

    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError("MSSSIM must have at least one example before it can be computed.")
        return self._sum / self._num_examples


class SSIM(ignite.metrics.Metric):
    def __init__(self, *, window_size=11, window=None, size_average=True, full=False, val_range=None, **super_kwargs):
        super().__init__(**super_kwargs)
        self.full = full
        self.size_average = size_average
        self.val_range = val_range
        self.window = window
        self.window_size = window_size

    def reset(self):
        self._sum = 0.0
        self._num_examples = 0

    def update(self, output):
        y_pred, y = output
        n = y.shape[0]
        self._sum += (
            ssim(
                y_pred.transpose(1, 2).flatten(end_dim=-4),
                y.transpose(1, 2).flatten(end_dim=-4),
                full=self.full,
                size_average=self.size_average,
                val_range=self.val_range,
                window=self.window,
                window_size=self.window_size,
            ).item()
            * n
        )
        self._num_examples += n

    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError("SSIM must have at least one example before it can be computed.")
        return self._sum / self._num_examples


# todo: msssim from skvideo for comparison
# class MSSSIM_SkVideo(Metric):
#     def __init__(self, window_size=11, size_average=True, val_range=None, normalize=False):
#             super().__init__(msssim_skvideo, window_size=window_size, size_average=size_average, val_range=val_range, normalize=normalize)


def ssim_skimage(pred, target, **kwargs):
    assert (
        len(target.shape) == 5 and pred.shape == target.shape
    ), f"Expecting nc(3 spacial dims), but got shapes {pred.shape}, {target.shape}"
    return numpy.mean(
        [
            [compare_ssim(pred[i, j], target[i, j], **kwargs) for j in range(target.shape[1])]
            for i in range(target.shape[0])
        ]
    )


class SSIM_SkImage(ignite.metrics.Metric):
    def __init__(
        self, *, win_size=11, gradient=False, data_range=None, gaussian_weights=False, full=False, **super_kwargs
    ):
        super().__init__(**super_kwargs)
        self.data_range = data_range
        self.full = full
        self.gaussian_weights = gaussian_weights
        self.gradient = gradient
        self.win_size = win_size

    def reset(self):
        self._sum = 0.0
        self._num_examples = 0

    def update(self, output):
        y_pred, y = output
        n = y.shape[0]
        self._sum += (
            ssim_skimage(
                y_pred.flatten(end_dim=-5).numpy(),
                y.flatten(end_dim=-5).numpy(),
                data_range=self.data_range,
                full=self.full,
                gaussian_weights=self.gaussian_weights,
                gradient=self.gradient,
                win_size=self.win_size,
            )
            * n
        )
        self._num_examples += n

    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError("SSIM_SKImage must have at least one example before it can be computed.")
        return self._sum / self._num_examples
