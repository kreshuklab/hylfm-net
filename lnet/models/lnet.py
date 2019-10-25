from typing import Tuple, Optional

from torch import nn


class LnetModel(nn.Module):
    @classmethod
    def get_scaling(cls, ipt_shape: Optional[Tuple[int, int]] = None) -> Tuple[float, float]:
        raise NotImplementedError

    @classmethod
    def get_shrinkage(cls, ipt_shape: Optional[Tuple[int, int]] = None) -> Tuple[int, int]:
        raise NotImplementedError

    def get_output_shape(self, ipt_shape: Tuple[int, int]):
        raise NotImplementedError
