import typing

import torch

from .base import LnetModel


class Dummy(LnetModel):
    def __init__(self, **kwargs):
        super().__init__()
        self.bias =  torch.nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, tensors: typing.OrderedDict[str, typing.Any]) -> typing.OrderedDict:
        return tensors

    # def get_scaling(self, ipt_shape: Optional[Tuple[int, int]] = None) -> Tuple[float, float]: