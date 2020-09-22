from __future__ import annotations

import collections.abc
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Union

import ignite.metrics
import numpy
import torch.nn

from hylfm.utils.general import camel_to_snake

logger = logging.getLogger(__name__)


@dataclass
class MetricValue:
    value: float
    higher_is_better: bool

    def as_float(self):
        if self.higher_is_better:
            return self.value
        else:
            return -self.value


class Metric(ignite.metrics.Metric):
    higher_is_better = True

    def __init__(self, *, postfix: str = "", tensor_names: Dict[str, str]):
        self.postfix = postfix
        self._required_output_keys = list(tensor_names.values())
        self.tensor_names = tensor_names
        super().__init__()

    @torch.no_grad()
    def iteration_completed(self, engine):
        output = self._output_transform(engine.state.output)
        assert isinstance(output, dict)
        output = self.prepare_for_update(output)
        assert isinstance(output, dict)
        self.update(output)

    def prepare_for_update(self, tensors: Dict) -> Dict:
        assert all([expected_name in tensors for expected_name in self.tensor_names.values()]), (
            self.tensor_names,
            tuple(tensors.keys()),
        )

        return tensors

    def update(self, tensors: Union[Tuple[Any, ...], Dict[str, Any]]) -> None:
        if isinstance(tensors, tuple):
            tensors = self.tensor_tuple_to_ordered_dict(tensors)

        tensors = self.prepare_for_update(tensors)
        self.update_impl(**{name: tensors[expected_name] for name, expected_name in self.tensor_names.items()})

    def tensor_tuple_to_ordered_dict(self, tensor_tuple: Tuple[Any, ...]):
        assert len(self._required_output_keys) == len(tensor_tuple), (self._required_output_keys, len(tensor_tuple))
        return collections.OrderedDict([(key, tensor) for key, tensor in zip(self._required_output_keys, tensor_tuple)])

    def completed(self, engine, name=""):
        result = self.compute()
        for subname, value in result.items():
            engine.state.metrics[name + subname] = value

    def value_to_metric_value(self, value: Union[float, List[float]]):
        if isinstance(value, list):
            return [self.value_to_metric_value(v) for v in value]
        else:
            assert isinstance(value, float)
            return MetricValue(value, self.higher_is_better)

    def compute(self) -> Dict[str, MetricValue]:
        computed = self.compute_impl()
        if isinstance(computed, MetricValue):
            return {camel_to_snake(self.__class__.__name__) + self.postfix: computed}
        elif isinstance(computed, float):
            return {camel_to_snake(self.__class__.__name__) + self.postfix: self.value_to_metric_value(computed)}
        elif isinstance(computed, dict):
            return {key + self.postfix: self.value_to_metric_value(value) for key, value in computed.items()}
        else:
            raise NotImplementedError(type(computed))

    @staticmethod
    def norm(unnormed: Union[numpy.ndarray, torch.Tensor]):
        return unnormed - unnormed.mean()

    def compute_impl(self) -> Union[MetricValue, float, Dict[str, Union[MetricValue, float]]]:
        raise NotImplementedError

    def update_impl(self, *args, **kwargs) -> None:
        raise NotImplementedError
