import typing

import torch

from .base import LnetModel


class Dummy(LnetModel):
    def __init__(
        self, input_name: str = "lfc", output_name: str = "pred", shrink: typing.Optional[int] = None
    ):
        super().__init__()
        self.input_name = input_name
        self.output_name = output_name
        self.shrink = shrink
        self.bias = torch.nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, tensors: typing.OrderedDict[str, typing.Any]) -> typing.OrderedDict:
        if self.shrink is None:
            tensors[self.output_name] = tensors[self.input_name]
        else:
            tensors[self.output_name] = tensors[self.input_name][
                ..., self.shrink : -self.shrink, self.shrink : -self.shrink
            ]

        for bmeta in tensors["meta"]:
            bmeta[self.output_name] = dict(bmeta[self.input_name])

        return tensors
