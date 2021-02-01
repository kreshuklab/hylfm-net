from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Union

import numpy
import torch.nn
from torch import no_grad

from hylfm.datasets import separate
from hylfm.hylfm_types import Array

logger = logging.getLogger(__name__)


class Metric:
    def __init__(
        self,
        *,
        per_sample: bool = True,
        along_dim: Optional[int] = None,
        dim_names: str = None,
        name: str = "{name}-{dim_name}",
        **super_kwargs,
    ):
        super().__init__(**super_kwargs)
        self.per_sample = per_sample
        self.along_dim = along_dim
        self.dim_names = dim_names
        self.dim_name = "" if along_dim is None else dim_names[along_dim]
        self.name = name.format(
            name=self.__class__.__name__.replace("_", "-"), **{k: str(v) for k, v in self.__dict__.items()}
        )
        self.reset()

    def reset(self):
        raise NotImplementedError

    def update_with_batch(self, **batch: Any) -> None:
        if not self.per_sample:
            raise NotImplementedError(f"compute on batch for {self.__class__}")

        if self.along_dim is None:
            for sample_in in separate(batch):
                self.update_with_sample(**sample_in)
        else:
            akey = next(iter(batch.keys()))
            for sample_in in separate(batch):
                assert all(isinstance(s, numpy.ndarray) or isinstance(s, torch.Tensor) for s in sample_in.values())
                len_ = sample_in[akey].shape[self.along_dim]
                assert all(s.shape[self.along_dim] == len_ for s in sample_in.values())
                for i in range(len_):
                    self.update_with_sample(
                        **{key: s[(slice(None),) * self.along_dim + (i,)] for key, s in sample_in.items()}
                    )

    def update_with_sample(self, **sample: Any) -> None:
        raise NotImplementedError

    def compute(self) -> Dict[str, Any]:
        raise NotImplementedError


class SimpleSingleValueMetric(Metric):
    _accumulated: float
    _n: int

    def reset(self):
        self._accumulated = 0.0
        self._n = 0

    @no_grad()
    def update_with_batch(self, prediction: Array, target: Array) -> None:
        value: Union[Array, float] = self(prediction, target)  # noqa
        if isinstance(value, (torch.Tensor, numpy.ndarray)):
            value = value.item()

        self._accumulated += value
        self._n += 1

    @no_grad()
    def update_with_sample(self, *args_sample, **sample: Any) -> None:
        self.update_with_batch(*(a[None] for a in args_sample), **{key: s[None] for key, s in sample.items()})

    def compute(self) -> Dict[str, Any]:
        return {self.name: self._accumulated / self._n}


class MetricGroup(Metric):
    def __init__(self, *metrics: Metric):
        self.metrics = metrics
        super().__init__()

    def reset(self):
        for metric in self.metrics:
            metric.reset()

    def update_with_batch(self, **batch: Any) -> None:
        for metric in self.metrics:
            metric.update_with_batch(**batch)

    def update_with_sample(self, **sample: Any) -> None:
        for metric in self.metrics:
            metric.update_with_sample(**sample)

    def compute(self) -> Dict[str, Any]:
        res = {}
        for metric in self.metrics:
            for key, value in metric.compute().items():
                assert key not in res, f"{key} already computed: {res[key]}"
                res[key] = value

        return res
