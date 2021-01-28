from typing import Optional, Union

import numpy
import torch

from hylfm.stat_ import DatasetStat
from .base import Transform
from ..hylfm_types import Array


class Normalize01Dataset(Transform):
    def __init__(
        self,
        *,
        min_: Optional[float] = None,
        max_: Optional[float] = None,
        min_percentile: Optional[float] = None,
        max_percentile: Optional[float] = None,
        clip: bool = False,
        apply_to: str,
    ):
        assert isinstance(apply_to, str)
        super().__init__(input_mapping={apply_to: "tensor", "stat": "stat"}, output_mapping={"tensor": apply_to})
        self.apply_to = apply_to

        if min_ is not None and min_percentile is not None:
            raise ValueError(f"exclusive arguments: use either `min_` or `min_percentile`")

        if max_ is not None and max_percentile is not None:
            raise ValueError(f"exclusive arguments: use either `max_` or `max_percentile`")

        if min_ is None and min_percentile is None:
            min_percentile = 0.0

        if max_ is None and max_percentile is None:
            max_percentile = 100.0

        self.min = min_
        self.min_percentile = min_percentile
        self.max = max_
        self.max_percentile = max_percentile
        self.clip = clip

    def apply_to_sample(self, tensor: Array, stat: DatasetStat):
        percentiles2compute = [
            p for p, m in [(self.min_percentile, self.min), (self.max_percentile, self.max)] if m is None
        ]
        if percentiles2compute:
            min_max = stat[self.apply_to].get_percentiles(name=self.apply_to, percentiles=percentiles2compute)
        else:
            min_max = []

        max_ = self.max or min_max.pop()
        min_ = self.min or min_max.pop()
        assert not min_max, min_max
        assert min_ < max_
        tensor = (tensor - min_) / (max_ - min_)
        if self.clip:
            tensor = numpy.clip(tensor, 0.0, 1.0)

        return tensor


class Normalize01Sample(Transform):
    def __init__(
        self,
        min_percentile: Optional[float] = None,
        max_percentile: Optional[float] = None,
        clip: bool = False,
        **super_kwargs,
    ):
        super().__init__(**super_kwargs)
        if min_percentile is None:
            min_percentile = 0.0

        if max_percentile is None:
            max_percentile = 100.0

        self.min_percentile = min_percentile
        self.max_percentile = max_percentile
        self.clip = clip

    def apply_to_sample(self, tensor: Array):
        if not isinstance(tensor, numpy.ndarray):
            raise NotImplementedError(type(tensor))

        if tensor.shape[0] != 1:
            raise NotImplementedError("multichannel")

        min_, max_ = numpy.percentile(
            tensor, [self.min_percentile, self.max_percentile]
        )  # todo multichannel with axis=0
        tensor = (tensor - min_) / (max_ - min_)
        if self.clip:
            tensor = numpy.clip(tensor, 0.0, 1.0)

        return tensor


class NormalizeMeanStd(Transform):
    def __init__(
        self,
        *,
        mean: Optional[float] = None,
        std: Optional[float] = None,
        percentile_min: Optional[float] = None,
        percentile_max: Optional[float] = None,
        determine_mean_std_from_stat: bool = False,
        epsilon: float = 1e-4,
        apply_to: str,
    ):
        assert isinstance(apply_to, str)
        super().__init__(input_mapping={apply_to: "tensor", "stat": "stat"}, output_mapping={"tensor": apply_to})
        self.apply_to = apply_to
        if mean is None and std is not None:
            raise ValueError("standard deviation `std` is specified, but `mean` is not!")

        if mean is not None and std is None:
            raise ValueError("mean is specified, but standard deviation `std` is not!")

        if percentile_min is not None or percentile_max is not None:
            if percentile_min is None:
                percentile_min = 0

            if percentile_max is None:
                percentile_max = 100.0

            percentile_range = (percentile_min, percentile_max)
        else:
            percentile_range = None

        if mean is not None and percentile_range is not None:
            raise ValueError(
                "exclusive arguments: use either `mean` and `std` or `percentile_min` and `percentile_max`"
            )

        if all(arg is None for arg in [mean, std, percentile_range]):
            percentile_range = (0.0, 100.0)

        self.percentile_range = percentile_range
        self.mean = mean
        self.std = std
        self.determine_mean_std_from_stat = determine_mean_std_from_stat
        self.epsilon = epsilon

    def apply_to_sample(self, tensor: Array, stat: DatasetStat):
        if self.mean is None:
            assert self.std is None
            if self.determine_mean_std_from_stat:
                mean, std = stat[self.apply_to].get_mean_std(name=self.apply_to, percentile_range=self.percentile_range)
            elif self.percentile_range == (0.0, 100.0):
                mean = tensor.mean()
                std = tensor.std()
            elif isinstance(tensor, numpy.ndarray):
                pmin, pmax = numpy.percentile(tensor, self.percentile_range)
                clipped_tensor = tensor.clip(pmin, pmax)
                mean = clipped_tensor.mean()
                std = clipped_tensor.std()
            else:
                raise NotImplementedError(type(tensor))
        else:
            assert self.std is not None
            mean, std = self.mean, self.std

        return (tensor - mean) / (std + self.epsilon)


class NormalizeMeanStdDataset(Transform):
    def __init__(
        self,
        *,
        mean: Optional[float] = None,
        std: Optional[float] = None,
        percentile_min: Optional[float] = None,
        percentile_max: Optional[float] = None,
        epsilon: float = 1e-7,
        apply_to: str,
    ):
        assert isinstance(apply_to, str)
        super().__init__(input_mapping={apply_to: "tensor", "stat": "stat"}, output_mapping={"tensor": apply_to})
        self.apply_to = apply_to

        if mean is None and std is not None:
            raise ValueError("standard deviation `std` is specified, but `mean` is not!")

        if mean is not None and std is None:
            raise ValueError("mean is specified, but standard deviation `std` is not!")

        if mean is not None and (percentile_min is not None or percentile_max is not None):
            raise ValueError(
                "exclusive arguments: use either mean and standard deviation std or a percentile_min/max "
                "to compute these. Default is percentile_min=0, percentile_max=100"
            )

        if mean is None:
            if percentile_min is None:
                percentile_min = 0

            if percentile_max is None:
                percentile_max = 100.0

        self.percentile_range = (percentile_min, percentile_max)
        self.mean = mean
        self.std = std
        self.epsilon = epsilon

    def apply_to_sample(self, tensor: Array, stat: DatasetStat):
        if self.mean is None:
            assert self.std is None
            mean, std = stat[self.apply_to].get_mean_std(name=self.apply_to, percentile_range=self.percentile_range)
        else:
            assert self.std is not None
            mean, std = self.mean, self.std

        return (tensor - mean) / (std + self.epsilon)


class NormalizeMeanStdSample(Transform):
    def __init__(
        self,
        percentile_min: Optional[float] = None,
        percentile_max: Optional[float] = None,
        epsilon: float = 1e-7,
        **super_kwargs,
    ):
        super().__init__(**super_kwargs)
        self.percentile_min = percentile_min
        self.percentile_max = percentile_max
        self.epsilon = epsilon

    def apply_to_sample(self, tensor: Array):
        if not isinstance(tensor, numpy.ndarray):
            raise NotImplementedError(type(tensor))

        if tensor.shape[0] != 1:
            raise NotImplementedError("multichannel")

        if self.percentile_min is None:
            min_ = None
        else:
            min_ = numpy.percentile(tensor, self.percentile_min)  # todo multichannel with axis=0

        if self.percentile_max is None:
            max_ = None
        else:
            max_ = numpy.percentile(tensor, self.percentile_max)  # todo multichannel with axis=0

        if min_ is not None or max_ is not None:
            tensor = tensor.clip(min=min_, max=max_)

        mean = tensor.mean()
        std = tensor.std()
        return (tensor - mean) / (std + self.epsilon)


class NormalizeMSE(Transform):
    def __init__(self, *, apply_to: str, target_name: str, return_alpha_beta: bool = True):
        assert isinstance(apply_to, str)
        super().__init__(input_mapping={apply_to: "ipt", "tgt": target_name}, output_mapping={"ipt": apply_to})
        self.apply_to = apply_to
        self.target_name = target_name
        self.return_alpha_beta = return_alpha_beta

    def apply_to_sample(self, ipt: Array, tgt: Array):
        if isinstance(ipt, torch.Tensor):
            ipt = ipt.cpu().numpy()

        ipt = ipt.astype(numpy.float32, copy=False)
        tgt = tgt.astype(numpy.float32, copy=False)

        cov = numpy.cov(ipt.flatten(), tgt.flatten())
        alpha = cov[0, 1] / (cov[0, 0] + 1e-10)
        beta = tgt.mean() - alpha * ipt.mean()

        ret = {"ipt": alpha * ipt + beta}
        if self.return_alpha_beta:
            prefix = f"{self.__class__.__name__}({self.apply_to},{self.target_name})."
            ret[prefix + "alpha"] = alpha
            ret[prefix + "beta"] = beta

        return ret
