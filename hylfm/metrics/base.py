from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

import numpy
import torch.nn
from torch import no_grad

from hylfm.hylfm_types import Array

logger = logging.getLogger(__name__)


class Metric:
    def __init__(
        self,
        *,
        per_sample: bool = False,
        along_dim: Optional[int] = None,
        dim_names: str = "czyx",
        name: str = "{name}-{dim_name}",
        **super_kwargs,
    ):
        assert along_dim is None or along_dim >= 0
        super().__init__(**super_kwargs)
        self.per_sample = per_sample
        self.along_dim = along_dim
        self.dim_names = dim_names
        self.dim_name = "" if along_dim is None else dim_names[along_dim]
        self.name = name.format(
            name=self.__class__.__name__.replace("_", "-"), **{k: str(v) for k, v in self.__dict__.items()}
        ).strip("-")
        assert not per_sample  # todo: remove per_sample
        self.reset()

    def reset(self):
        raise NotImplementedError(self)

    def update_with_batch(self, prediction: Array, target: Array) -> Dict[str, Any]:
        batch = iter(zip(prediction, target))
        pred, tgt = next(batch)
        ret = {k: [v] for k, v in self.update_with_sample(prediction=pred, target=tgt).items()}
        for pred, tgt in batch:
            for k, v in self.update_with_sample(prediction=pred, target=tgt).items():
                assert k in ret
                ret[k].append(v)

        return ret

        # if not self.per_sample:
        #     raise NotImplementedError(f"compute on batch for {self.__class__}")
        #
        # if self.along_dim is None:
        #     ret = [self.update_with_sample(**sample_in) for sample_in in separate(batch)]
        # else:
        #     ret = []
        #     akey = next(iter(batch.keys()))
        #     for sample_in in separate(batch):
        #         assert all(isinstance(s, numpy.ndarray) or isinstance(s, torch.Tensor) for s in sample_in.values())
        #         len_ = sample_in[akey].shape[self.along_dim]
        #         assert all(s.shape[self.along_dim] == len_ for s in sample_in.values())
        #         ret.append(
        #             [
        #                 self.update_with_sample(
        #                     **{key: s[(slice(None),) * self.along_dim + (i,)] for key, s in sample_in.items()}
        #                 )
        #                 for i in range(len_)
        #             ]
        #         )
        #
        # return ret

    def update_with_sample(self, **sample: Any) -> Dict[str, Any]:
        raise NotImplementedError

    def compute(self) -> Dict[str, Any]:
        raise NotImplementedError


class SimpleSingleValueMetric(Metric):
    _accumulated: Optional[Union[float, numpy.ndarray]]
    _n: int

    def reset(self):
        self._accumulated = None
        self._n = 0

    @no_grad()
    def update_with_sample(self, prediction: Array, target: Array) -> Dict[str, Any]:
        if self.along_dim is None:
            value: Union[Array, float] = self(prediction[None], target[None])  # noqa
            if isinstance(value, (torch.Tensor, numpy.ndarray)):
                value = value.item()

        else:
            dim_len = target.shape[self.along_dim]
            if self._accumulated is None:
                self._accumulated = numpy.zeros(dim_len, dtype=numpy.float32)

            value = numpy.empty(dim_len, dtype=numpy.float32)
            for d in range(dim_len):
                slice_tuple = (slice(None),) * self.along_dim + (d,)
                val = self(prediction[slice_tuple][None], target[slice_tuple][None])  # noqa
                if isinstance(val, (torch.Tensor, numpy.ndarray)):
                    val = val.item()

                value[d] = val

        self._n += 1
        if self._accumulated is None:
            self._accumulated = value
        else:
            self._accumulated += value

        return {self.name: value}

    def compute(self) -> Dict[str, Any]:
        return {self.name: self._accumulated / self._n}


class MetricGroup(Metric):
    def __init__(self, *metrics: Metric):
        self.metrics = metrics
        super().__init__()

    def reset(self):
        for metric in self.metrics:
            metric.reset()

    def update_with_batch(self, **batch: Any) -> Dict[str, Any]:
        res = {}
        for metric in self.metrics:
            for key, val in metric.update_with_batch(**batch).items():
                assert key not in res, key
                assert isinstance(val, list), key
                res[key] = val

        return res

    def update_with_sample(self, **sample: Any) -> None:
        raise NotImplementedError

    def compute(self) -> Dict[str, Any]:
        res = {}
        for metric in self.metrics:
            for key, value in metric.compute().items():
                assert key not in res, f"{key} already computed: {res[key]}"
                res[key] = value

        return res

    def __add__(self, other: Metric):
        if isinstance(other, MetricGroup):
            return MetricGroup(*(list(self.metrics) + list(other.metrics)))
        elif isinstance(other, Metric):
            return MetricGroup(*(list(self.metrics) + [other]))
        else:
            raise NotImplementedError(type(other))
