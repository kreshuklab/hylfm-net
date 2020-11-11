from typing import Any, Dict, List, Optional, Type, Union

import torch
from ignite.exceptions import NotComputableError

from hylfm.losses.on_tensors import LossOnTensors
from hylfm.metrics.scale_minimize_vs import ScaleMinimizeVsMetric


class MetricFromLoss(ScaleMinimizeVsMetric):
    higher_is_better = False

    def __init__(
        self,
        *super_args,
        tensor_names: Union[List[str], Dict[str, str]],
        loss_class: Type[LossOnTensors],
        loss_kwargs: Dict[str, Any],
        **super_kwargs,
    ):
        if isinstance(tensor_names, dict):
            metric_tensor_names = {expected_name: expected_name for expected_name in tensor_names.values()}
        else:
            metric_tensor_names = {expected_name: expected_name for expected_name in tensor_names}

        super().__init__(*super_args, tensor_names=metric_tensor_names, **super_kwargs)
        self.loss = loss_class(tensor_names=tensor_names, **loss_kwargs)
        if self.loss.is_inverted_metric:
            self.higher_is_better = True

    def reset(self):
        self._accumulated = 0.0
        self._n = 0

    def update_impl(self, **tensors) -> None:
        batch_len = len(next(iter(tensors.values())))
        assert batch_len > 0
        self._n += batch_len
        with torch.no_grad():
            self.loss(tensors)
            self._accumulated += float(tensors[self.loss.name].item()) * batch_len

    def compute_impl(self):
        loss_name = self.loss.name
        if not self._n:
            raise NotComputableError(
                f"{self.__class__.__name__} must have at least one example before it can be computed."
            )
        value = self._accumulated / self._n
        if self.loss.is_inverted_metric:
            value *= -1

        return {loss_name: value}
