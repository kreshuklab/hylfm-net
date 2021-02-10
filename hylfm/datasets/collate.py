from functools import partial
from itertools import chain
from typing import Any, Collection, Dict, List, Optional, Union

import numpy
import torch

from hylfm.hylfm_types import TransformLike


COMMON_BATCH_KEYS = {"batch_len", "epoch", "epoch_len", "iteration"}  # are shared across all samples in a batch
SAMPLE_KEYS_EQUAL_IN_BATCH = {"crop_name"}  # each sample has it, but they need to equal to be batched together


def sample_values_to_batch_value(values: List, *, sample_key: Optional = None):
    v0 = values[0]

    if isinstance(v0, numpy.ndarray):
        batch_value = numpy.ascontiguousarray(numpy.stack(values, axis=0))
    elif isinstance(v0, torch.Tensor):
        batch_value = torch.stack(values, dim=0)
    elif sample_key in COMMON_BATCH_KEYS:
        raise ValueError(f"invalid key in sample: {sample_key}")
    elif sample_key in SAMPLE_KEYS_EQUAL_IN_BATCH:
        batch_value = v0
        assert all(v0 == v for v in values)
    else:
        batch_value = values

    return batch_value


def collate(tensors: List[Dict[str, Any]]) -> Dict[str, Any]:
    assert tensors
    if "batch_len" in tensors[0]:
        assert all("batch_len" in d for d in tensors)
        return collate_batches(tensors)
    else:
        assert not any("batch_len" in d for d in tensors)
        return collate_samples(tensors)


def verify_keys(tensors: List[Dict[str, Any]], name: str) -> Collection[str]:
    assert len(tensors) > 0
    keys = set(tensors[0].keys())
    different_keys = [set(b.keys()) for b in tensors if set(b.keys()) != keys]
    if different_keys:
        raise KeyError(f"Expected all {name} to have keys: {keys}, but found different keys: {different_keys}")

    return keys


def condense_common_values(values: List[Any], key: str) -> Any:
    assert values
    if key == "batch_len":
        return sum(values)
    else:  # all equal
        value = values[0]
        assert all(v == value for v in values[1:]), values
        return value


def collate_batches(batches: List[Dict[str, Any]]) -> Dict[str, Any]:
    keys = verify_keys(batches, "batches")
    listed_batches = {key: [b[key] for b in batches] for key in keys}
    batch = {
        key: condense_common_values(batch_values, key)
        if key in COMMON_BATCH_KEYS or key in SAMPLE_KEYS_EQUAL_IN_BATCH
        else stack_batch_values(batch_values)
        for key, batch_values in listed_batches.items()
    }
    return batch


def stack_batch_values(batch_values: Union[list, numpy.ndarray, torch.Tensor]):
    assert batch_values
    if isinstance(batch_values[0], list):
        return sum(batch_values, [])
    elif isinstance(batch_values[0], numpy.ndarray):
        return numpy.concatenate(batch_values, axis=0)
    elif isinstance(batch_values[0], torch.Tensor):
        return torch.cat(batch_values, dim=0)
    else:
        raise TypeError(type(batch_values[0]))


def collate_samples(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    keys = verify_keys(samples, "samples")
    batch = {key: sample_values_to_batch_value([s[key] for s in samples], sample_key=key) for key in keys}
    batch["batch_len"] = len(samples)
    return batch


def collate_and_batch_transform(samples, *, transform: TransformLike):
    batch = collate(samples)
    return transform(batch)


def batch_value_to_sample_values(value: Any, *, sample_key: Optional = None) -> list:
    if isinstance(value, (numpy.ndarray, torch.Tensor)):
        sample_values = list(value)
    elif isinstance(value, list):
        sample_values = value
    elif sample_key in COMMON_BATCH_KEYS:
        raise ValueError(f"invalid sample key: {sample_key}")
    elif sample_key in SAMPLE_KEYS_EQUAL_IN_BATCH:
        raise ValueError("tread SAMPLE_KEYS_EQUAL_IN_BATCH as COMMON_BATCH_KEYS here")
    else:
        raise NotImplementedError(f"{type(value)} for sample_key: {sample_key}")

    return sample_values


def get_collate(batch_transformation: TransformLike):
    return partial(collate_and_batch_transform, transform=batch_transformation)


def separate(batch: Dict[str, Any]) -> List[Dict[str, Any]]:
    batch = dict(batch)
    batch_len = batch.pop("batch_len", None)

    batch = dict(batch)
    common = {k: batch.pop(k) for k in chain(COMMON_BATCH_KEYS, SAMPLE_KEYS_EQUAL_IN_BATCH) if k in batch}

    samples = [
        dict(zip(batch, sv), **common)
        for sv in zip(*(batch_value_to_sample_values(bv, sample_key=k) for k, bv in batch.items()))
    ]
    assert batch_len is None or len(samples) == batch_len
    return samples
