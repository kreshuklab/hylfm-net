from typing import Any, Dict, Union

import numpy
import torch

try:
    from typing import Protocol
except ImportError:
    from typing_extensions import Protocol


class TransformLike(Protocol):
    def __call__(self, tensors: Dict[str, Any]) -> Dict[str, Any]:
        pass


Array = Union[numpy.ndarray, torch.Tensor]
