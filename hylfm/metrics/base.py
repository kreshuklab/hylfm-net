import collections
import logging
from abc import abstractmethod
from typing import Any, Dict, Optional, OrderedDict, Tuple, Union

import ignite.metrics
import numpy
import torch.nn

from hylfm.metrics.scale_minimize_vs import ScaleMinimizeVsMixin
from hylfm.utils.general import camel_to_snake

logger = logging.getLogger(__name__)


class Metric(ScaleMinimizeVsMixin, ignite.metrics.Metric):
    def __init__(
        self,
        *,
        postfix: str = "",
        tensor_names: Dict[str, str],
        scale_minimize_vs: Optional[Tuple[str, str, str]] = None,
    ):
        self.postfix = postfix
        self._required_output_keys = list(tensor_names.keys())
        super().__init__(tensor_names=tensor_names, scale_minimize_vs=scale_minimize_vs)

    def prepare_for_update(self, tensors: OrderedDict) -> OrderedDict:
        for unnormed, normed in self.map_unnormed.items():
            if normed not in tensors:
                tensors[normed] = self.norm(tensors[unnormed])

        for unscaled, scaled in self.map_unscaled.items():
            if scaled not in tensors:
                tensors[scaled] = self.scale(tensors[unscaled], tensors[self.vs_norm])

        assert all([expected_name in tensors for expected_name in self.tensor_names.values()]), (
            self.tensor_names,
            tuple(tensors.keys()),
        )

        return tensors

    def update(self, tensors: Union[Tuple[Any, ...], OrderedDict[str, Any]]) -> None:
        if isinstance(tensors, tuple):
            tensors = self.tensor_tuple_to_ordered_dict(tensors)

        tensors = self.prepare_for_update(tensors)
        self.update_impl(**{name: tensors[expected_name] for name, expected_name in self.tensor_names.items()})

    def tensor_tuple_to_ordered_dict(self, tensor_tuple: Tuple[Any, ...]):
        assert len(self._required_output_keys) == len(tensor_tuple), (self._required_output_keys, len(tensor_tuple))
        return collections.OrderedDict([(key, tensor) for key, tensor in zip(self._required_output_keys, tensor_tuple)])

    def compute(self) -> Dict[str, float]:
        computed = self.compute_impl()
        if isinstance(computed, float):
            return {camel_to_snake(self.__class__.__name__) + self.postfix: float(computed)}
        elif isinstance(computed, (dict, OrderedDict)):
            return {key + self.postfix: value for key, value in computed.items()}
        else:
            raise NotImplementedError(type(computed))

    @staticmethod
    def norm(unnormed: Union[numpy.ndarray, torch.Tensor]):
        return unnormed - unnormed.mean()

    @abstractmethod
    def compute_impl(self) -> Union[float, Dict[str, float]]:
        raise NotImplementedError

    @abstractmethod
    def update_impl(self, **kwargs) -> None:
        raise NotImplementedError
