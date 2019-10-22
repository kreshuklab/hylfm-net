from typing import Callable

import torch
from ignite.exceptions import NotComputableError
from ignite.metrics import Metric

from lnet.output import Output


class OutputMetric(Metric):
    """
    Compute average loss from output with loss attribute
    """

    def __init__(self, out_to_metric: Callable[[Output], torch.Tensor]):
        def output_trf(out: Output):
            return out_to_metric(out), out.tgt.shape[0]

        super().__init__(output_trf)

    def reset(self):
        self._sum = 0
        self._num_examples = 0

    def update(self, transformed_output):
        batch_value, batch_size = transformed_output

        self._sum += batch_value.item() * batch_size
        self._num_examples += batch_size

    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError(
                'Loss must have at least one example before it can be computed.')

        return self._sum / self._num_examples
