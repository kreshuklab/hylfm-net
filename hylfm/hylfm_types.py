from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, NamedTuple, Optional, TYPE_CHECKING, Union

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


Array = Union[numpy.ndarray, torch.Tensor]


class CriterionLike(Protocol):
    minimize: bool

    def __call__(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        *,
        epoch: Optional[int],
        iteration: Optional[int],
        epoch_len: Optional[int]
    ) -> torch.Tensor:
        pass


class TransformLike(Protocol):
    def __call__(self, tensors: Dict[str, Any]) -> Dict[str, Any]:
        pass


class CriterionChoice(str, Enum):
    L1 = "L1"
    MS_SSIM = "MS_SSIM"
    MSE = "MSE"
    SmoothL1 = "SmoothL1"
    SmoothL1_MS_SSIM = "SmoothL1_MS_SSIM"
    WeightedL1 = "WeightedL1"
    WeightedSmoothL1 = "WeightedSmoothL1"
    WeightedL1_MS_SSIM = "WeightedL1_MS_SSIM"
    WeightedSmoothL1_MS_SSIM = "WeightedSmoothL1_MS_SSIM"


class DatasetAndTransforms(NamedTuple):
    dataset: "ConcatDataset"
    batch_preprocessing: TransformLike
    batch_preprocessing_in_step: TransformLike
    batch_postprocessing: TransformLike


class DatasetChoice(str, Enum):
    beads_sample0 = "beads_sample0"
    beads_highc_a = "beads_highc_a"
    beads_highc_b = "beads_highc_b"
    heart_static_sample0 = "heart_static_sample0"
    heart_static_a = "heart_static_a"
    heart_static_b = "heart_static_b"
    heart_static_c = "heart_static_c"  # a with 99.8 spim percentile norm
    # heart_static_c_care = "heart_static_c_care"  # test data like heart_static_c, but also load lr and care
    heart_static_c_care_complex = "heart_static_c_care_complex"
    heart_static_fish2_f4 = "heart_static_fish2_f4"
    heart_static_fish2_f4_sliced = "heart_static_fish2_f4_sliced"


class DatasetPart(str, Enum):
    train = "train"
    validate = "validate"
    test = "test"


class OptimizerChoice(str, Enum):
    Adam = "Adam"
    SGD = "SGD"


class PeriodUnit(str, Enum):
    epoch = "epoch"
    iteration = "iteration"


@dataclass
class TransformsPipeline:
    sample_precache_trf: List[Dict[str, Dict[str, Any]]]
    sample_preprocessing: TransformLike
    batch_preprocessing: TransformLike
    batch_preprocessing_in_step: TransformLike
    batch_postprocessing: TransformLike
    batch_premetric_trf: TransformLike
    meta: Dict[str, Any]
    tgt_name: str
    spatial_dims: int
