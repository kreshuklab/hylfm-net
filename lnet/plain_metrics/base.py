import typing
from collections import OrderedDict

import torch.nn

from lnet.utils.general import camel_to_snake


class Metric:
    def __init__(self, *, postfix: str = "", **tensor_names):
        self.tensor_names = tensor_names
        self.postfix = postfix
        self.reset()

    def reset(self):
        raise NotImplementedError

    def update(self, tensors: typing.OrderedDict) -> None:
        assert all([expected_name in tensors for expected_name in self.tensor_names.values()]), (
            self.tensor_names,
            list(tensors.keys()),
        )
        self.update_impl(**{name: tensors[expected_name] for name, expected_name in self.tensor_names.items()})

    def update_impl(self, **kwargs) -> None:
        raise NotImplementedError

    def compute(self) -> typing.Dict[str, float]:
        computed = self.compute_impl()
        if isinstance(computed, float):
            return {camel_to_snake(self.__class__.__name__) + self.postfix: float(computed)}
        elif isinstance(computed, (dict, OrderedDict)):
            return {key + self.postfix: float(value) for key, value in computed.items()}
        else:
            raise NotImplementedError(type(computed))

    def compute_impl(self) -> typing.Union[float, typing.Dict[str, float]]:
        raise NotImplementedError


class LossAsMetric(Metric):
    def __init__(self, loss: torch.nn.Module, *, postfix: str = ""):
        """note: override 'update_impl if custom_tensor_names need to be specified"""
        self.loss = loss
        super().__init__(postfix=postfix)

    def reset(self):
        self._accumulated = 0.0
        self._n = 0

    def update(self, tensors: typing.OrderedDict[str, typing.Any]) -> None:
        self._n += len(tensors["meta"])
        with torch.no_grad():
            self._accumulated += float(self.loss(tensors).item())

    def compute_impl(self):
        loss_name = camel_to_snake(self.loss.__class__.__name__)
        return {loss_name: self._accumulated / self._n}
