from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, NamedTuple, TYPE_CHECKING, Union

import numpy
import torch

if TYPE_CHECKING:
    from hylfm.datasets import ConcatDataset

try:
    from typing import Protocol
except ImportError:
    from typing_extensions import Protocol

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


class TransformLike(Protocol):
    def __call__(self, tensors: Dict[str, Any]) -> Dict[str, Any]:
        pass


class CriterionLike(Protocol):
    def __call__(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pass


Array = Union[numpy.ndarray, torch.Tensor]


class DatasetName(str, Enum):
    beads_sample0 = "beads_sample0"
    beads_highc_a = "beads_highc_a"
    beads_highc_b = "beads_highc_b"
    heart_static_a = "heart_static_a"


class DatasetPart(str, Enum):
    train = "train"
    validate = "validate"
    test = "test"


class DatasetAndTransforms(NamedTuple):
    dataset: "ConcatDataset"
    batch_preprocessing: TransformLike
    batch_preprocessing_in_step: TransformLike
    batch_postprocessing: TransformLike


@dataclass
class TransformsPipeline:
    sample_precache_trf: List[Dict[str, Dict[str, Any]]]
    sample_preprocessing: TransformLike
    batch_preprocessing: TransformLike
    batch_preprocessing_in_step: TransformLike
    batch_postprocessing: TransformLike
    batch_premetric_trf: TransformLike
    meta: Dict[str, Any]
