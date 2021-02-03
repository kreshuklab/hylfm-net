from math import log10

import numpy
import torch.nn.functional

from hylfm.metrics import SimpleSingleValueMetric


# class PSNR_SkImage(Metric):
#     sum_: float
#     num_examples: int
#
#     def __init__(self, *, data_range=None, **super_kwargs):
#         super().__init__(**super_kwargs)
#         self.data_range = data_range
#
#     def reset(self):
#         self.sum_ = 0.0
#         self.num_examples = 0
#
#     def update_with_batch(self, *, prediction, target) -> None:
#         n = prediction.shape[0]
#         self.sum_ += sum(
#             compare_psnr(im_test=p, im_true=t, data_range=self.data_range)
#             for p, t in zip(prediction.cpu().numpy(), target.cpu().numpy())
#         )
#         self.num_examples += n
#
#     update_with_sample = update_with_batch
#
#     def compute(self):
#         if self.num_examples == 0:
#             raise RuntimeError("PSNR_SKImage must have at least one example before it can be computed.")
#
#         return {self.name: self.sum_ / self.num_examples}


class PSNR(SimpleSingleValueMetric):
    def __init__(self, *super_args, data_range: float, **super_kwargs):
        super().__init__(*super_args, **super_kwargs)
        self.log10dr20 = log10(data_range) * 20

    @torch.no_grad()
    def __call__(self, prediction, target):
        n = prediction.shape[0]
        prediction = torch.from_numpy(prediction) if isinstance(prediction, numpy.ndarray) else prediction
        target = torch.from_numpy(target) if isinstance(target, numpy.ndarray) else target

        return (
            sum(
                self.log10dr20 - 10 * torch.log10(torch.nn.functional.mse_loss(p, t)).item()
                for p, t in zip(prediction, target)
            )
            / n
        )
