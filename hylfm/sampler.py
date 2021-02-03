import logging
from collections import defaultdict
from typing import DefaultDict, List, Type

import numpy
import torch.utils.data.sampler
from torch.utils.data import ConcatDataset

logger = logging.getLogger(__name__)


class NoCrossBatchSampler(torch.utils.data.sampler.BatchSampler):
    """Wraps another sampler to yield a mini-batch of indices,
        while never mixing mini batches across datasets of the given (ConcatDataset's) cumulative indices.

    Args:
        cumulative_indices (sequence): cumulative indices to not batch across.
        sampler (Sampler): Model sampler.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``

    """

    def __init__(
        self,
        concat_dataset: ConcatDataset,
        sampler_class: Type[torch.utils.data.sampler.Sampler],
        batch_sizes: List[int],
        drop_last: bool,
    ):
        if not issubclass(sampler_class, torch.utils.data.sampler.Sampler):
            raise ValueError(f"sampler_class should inherite from torch.utils.data.Sampler, but got {sampler_class}")
        if (
            not isinstance(batch_sizes, list)
            or len(batch_sizes) != len(concat_dataset.datasets)
            or any(bs <= 0 for bs in batch_sizes)
        ):
            raise ValueError(
                f"batch_sizes should be a list of positive integer values with the same length as concat_dataset.concat_datasets, but got batch_sizes={batch_sizes}"
            )
        if not isinstance(drop_last, bool):
            raise ValueError(f"drop_last should be a boolean value, but got drop_last={drop_last}")

        self.sampler = sampler_class(concat_dataset)
        self.batch_sizes = batch_sizes
        self.drop_last = drop_last

        logger.warning("assuming that `len(datasource)=len(sampler_class(datasource))`!")
        self._len = 0
        for i, ds in enumerate(concat_dataset.datasets):
            if self.drop_last:
                self._len += len(ds) // self.batch_sizes[i]
            else:
                self._len += (len(ds) + self.batch_sizes[i] - 1) // self.batch_sizes[i]

        self.cumsum = concat_dataset.cumulative_sizes

    def __iter__(self):
        batches: DefaultDict[int, List[int]] = defaultdict(list)
        for idx in self.sampler:
            ds = numpy.searchsorted(self.cumsum, idx, side="right")
            batches[ds].append(idx)
            if len(batches[ds]) == self.batch_sizes[ds]:
                yield batches.pop(ds)

        if batches and not self.drop_last:
            for batch in batches.values():
                yield batch

    def __len__(self):
        return self._len
