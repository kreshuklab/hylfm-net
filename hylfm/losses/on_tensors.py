from __future__ import annotations

import logging
from typing import Any, Dict

from hylfm.utils.general import camel_to_snake

logger = logging.getLogger(__name__)


class LossOnTensors:
    is_inverted_metric = False

    def __init__(self, *super_args, tensor_names: Dict[str, str], prefix: str = "", **super_kwargs):
        self.tensor_names = tensor_names
        self.name = camel_to_snake(prefix + self.__class__.__name__)
        super().__init__(*super_args, **super_kwargs)


class LossOnTensorsTorchMixin(LossOnTensors):
    def __call__(self, tensors: Dict[str, Any]) -> None:
        try:
            loss_value = super().__call__(
                **{name: tensors[expected_name] for name, expected_name in self.tensor_names.items()}
            )

            if isinstance(loss_value, dict):
                assert self.name in loss_value
                for k in loss_value.keys():
                    assert k not in tensors, f"{k} already in tensors: {list(tensors.keys())}"

                tensors.update(loss_value)
            else:
                assert self.name not in tensors, f"{self.name} already in tensors: {list(tensors.keys())}"
                tensors[self.name] = loss_value

        except Exception as e:
            logger.error("Could not call %s", self)
            raise e

    def forward(self, prediction, target):
        value = super().forward(prediction, target)
        if self.is_inverted_metric:
            return -value
        else:
            return value
