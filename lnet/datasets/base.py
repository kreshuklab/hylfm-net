import logging
import re
import threading
import typing
from collections import OrderedDict
from concurrent.futures import Future
from concurrent.futures.thread import ThreadPoolExecutor
from hashlib import sha224 as hash_algorithm
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import h5py
import imageio
import numpy
import torch.utils.data
import torch.multiprocessing
import warnings
import yaml
import z5py

import lnet
from lnet import settings
from lnet.datasets.utils import get_paths_and_numbers
from lnet.stat import DatasetStat
from lnet.transformations.base import ComposedTransform

logger = logging.getLogger(__name__)

GKRESHUK = settings.data_roots.GKRESHUK
GHUFNAGELLFLenseLeNet_Microscope = settings.data_roots.GHUFNAGELLFLenseLeNet_Microscope


class PathOfInterest:
    def __init__(self, *points: Tuple[int, int, int, int], sigma: int = 1):
        self.points = points
        self.sigma = sigma


class TensorInfo:
    def __init__(
        self,
        *,
        name: str,
        root: str,
        location: str,
        transformations: Sequence[Dict[str, Any]] = tuple(),
        in_batches_of: int = 1,
        insert_singleton_axes_at: Sequence[int] = tuple(),
        z_slice: Optional[Union[str, int]] = None,
        skip_indices: Sequence[int] = tuple(),
        meta: Optional[dict] = None,
        **kwargs,
    ):
        if z_slice is not None and skip_indices:
            raise NotImplementedError("skip indices with z_slice")

        assert isinstance(name, str)
        assert isinstance(root, str)
        assert isinstance(location, str)
        assert isinstance(in_batches_of, int)
        self.name = name
        self.root = root
        self.transformations = list(transformations)
        self.in_batches_of = in_batches_of
        self.insert_singleton_axes_at = insert_singleton_axes_at
        self.z_slice = z_slice
        self.skip_indices = skip_indices
        self.meta: dict = meta or {}
        self.kwargs = kwargs
        self.location = location
        self.path: Path = getattr(settings.data_roots, root) / location

    @property
    def transformations(self) -> List[Dict[str, Any]]:
        return self.__transformations

    @transformations.setter
    def transformations(self, trfs):
        assert isinstance(trfs, list)
        trfs = [{name: kwargs for name, kwargs in trf.items() if self.name in kwargs["apply_to"]} for trf in trfs]
        discarded_trfs = [
            {name: kwargs for name, kwargs in trf.items() if self.name not in kwargs["apply_to"]} for trf in trfs
        ]
        for dtrf in discarded_trfs:
            for name, kwargs in dtrf:
                warnings.warn(f"discarded trf {name} for tensor {self.name} (apply_to: {kwargs['apply_to']})")

        assert not any([getattr(lnet.transformations, name).randomly_changes_shape for trf in trfs for name in trf])
        self.__transformations = trfs

    @property
    def description(self):
        return yaml.safe_dump(
            {
                "name": self.name,
                "root": self.root,
                "location": self.location,
                "transformations": self.transformations,
                "in_batches_of": self.in_batches_of,
                "insert_singleton_axes_at": self.insert_singleton_axes_at,
                "z_slice": self.z_slice,
                "skip_indices": list(self.skip_indices),
                "meta": self.meta,
                "kwargs": self.kwargs,
            }
        )


class DatasetFromInfo(torch.utils.data.Dataset):
    get_z_slice: Callable[[int], Optional[int]]

    def __init__(self, *, info: TensorInfo):
        super().__init__()
        self.tensor_name = info.name
        self.description = info.description
        self.transform = lnet.transformations.ComposedTransform(
            *[
                getattr(lnet.transformations, name)(**kwargs)
                for trf in info.transformations
                for name, kwargs in trf.items()
            ]
        )

        self.in_batches_of = info.in_batches_of
        self.insert_singleton_axes_at = info.insert_singleton_axes_at

        self._z_slice_mod: Optional[int] = None
        if info.z_slice is None:
            self._z_slice = None
        elif isinstance(info.z_slice, int):
            self._z_slice = info.z_slice
        elif isinstance(info.z_slice, str):
            if info.z_slice.startswith("idx%"):
                self._z_slice_mod = int(info.z_slice[4:])
            else:
                raise NotImplementedError(info.z_slice)
        else:
            raise NotImplementedError(info.z_slice)

    def get_z_slice(self, idx: int) -> int:
        if self._z_slice is None:
            return None
        elif self._z_slice_mod is None:
            return self.z_slice
        else:
            return idx % self._z_slice_mod

    def update_meta(self, meta: dict) -> dict:
        has_z_slice = meta.get("z_slice", None)
        z_slice = self.get_z_slice(meta["idx"])
        if z_slice is not None:
            assert has_z_slice is None or has_z_slice == z_slice
            meta["z_slice"] = z_slice

        return meta

    def shutdown(self):
        pass


class TiffDataset(DatasetFromInfo):
    def __init__(self, *, info: TensorInfo):
        assert not info.kwargs, info.kwargs
        super().__init__(info=info)
        paths, numbers = get_paths_and_numbers(info.path)
        self.paths = [p for i, p in enumerate(paths) if i not in info.skip_indices]
        self.numbers = numbers

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int) -> typing.OrderedDict[str, Union[numpy.ndarray, list]]:
        path_idx = idx // self.in_batches_of
        idx %= self.in_batches_of
        img_path = self.paths[path_idx]
        img: numpy.ndarray = imageio.volread(img_path)
        for axis in self.insert_singleton_axes_at:
            img = numpy.expand_dims(img, axis=axis)

        return self.transform(OrderedDict(**{self.tensor_name: img[idx : idx + 1]}))


class H5Dataset(DatasetFromInfo):
    @staticmethod
    def get_ds_resolver(pattern: str, results_to: List[str]) -> Callable[[str], None]:
        def ds_resolver(name: str):
            if re.match(pattern, name):
                results_to.append(name)

        return ds_resolver

    def __init__(self, *, info: TensorInfo):
        if info.skip_indices:
            raise NotImplementedError("skip_indices")

        if info.kwargs:
            raise NotImplementedError(info.kwargs)
        super().__init__(info=info)
        h5_ext = ".h5"
        assert h5_ext in info.location.as_posix(), info.location.as_posix()
        file_path_glob, within_pattern = info.location.as_posix().split(h5_ext)
        within_pattern = within_pattern.strip("/")
        file_path_glob += h5_ext
        paths, numbers = get_paths_and_numbers(Path(file_path_glob))
        self.numbers = numbers
        self.h5files = [h5py.File(p, mode="r") for p in paths]
        self.within_paths = []
        for hf in self.h5files:
            root_group = hf["/"]
            within = []
            root_group.visit(self.get_ds_resolver(within_pattern, within))
            self.within_paths.append(sorted(within))

        assert all(self.within_paths), self.within_paths
        self._shutdown = False

    def __len__(self):
        return len(self.h5files)

    def __getitem__(self, idx: int) -> typing.OrderedDict[str, Union[numpy.ndarray, list]]:
        assert not self._shutdown
        path_idx = idx // self.in_batches_of
        idx %= self.in_batches_of
        hf = self.h5files[path_idx]
        withins = self.within_paths[path_idx]
        h5ds = hf[withins[idx]]
        img: numpy.ndarray = h5ds[:]
        for axis in self.insert_singleton_axes_at:
            img = numpy.expand_dims(img, axis=axis)

        return self.transform(OrderedDict(**{self.tensor_name: img[idx : idx + 1]}))

    def shutdown(self):
        self._shutdown = True
        [hf.close() for hf in self.h5files]


def get_dataset_from_info(info: TensorInfo) -> DatasetFromInfo:
    if str(info.location).endswith(".tif"):
        return TiffDataset(info=info)
    elif ".h5" in str(info.location):
        return H5Dataset(info=info)
    else:
        raise NotImplementedError


N5CachedDataset_submit_lock = torch.multiprocessing.Lock()


class N5CachedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: DatasetFromInfo):
        super().__init__()
        self.dataset = dataset
        data_cache_path = settings.data_roots.lnet / "data"
        assert data_cache_path.exists(), data_cache_path.absolute()
        description = dataset.description

        data_file_path = data_cache_path / f"{hash_algorithm(description.encode()).hexdigest()}.n5"
        data_file_path.with_suffix(".txt").write_text(description)

        logger.info("cache %s to %s", dataset.tensor_name, data_file_path)
        self.tensor_name = tensor_name = dataset.tensor_name

        self.data_file = data_file = z5py.File(path=str(data_file_path), mode="a", use_zarr_format=False)
        shape = data_file[tensor_name].shape if tensor_name in data_file else None

        if shape is None:
            self._len = _len = len(dataset)
            sample = dataset[0]
            tensor = sample[tensor_name]
            tensor_shape = tuple(tensor.shape)
            assert tensor_shape[0] == 1, tensor_shape  # expected explicit batch dimension
            shape = (_len,) + tensor_shape[1:]
            data_file.create_dataset(tensor_name, shape=shape, chunks=tensor_shape, dtype=tensor.dtype)
        else:
            self._len = _len = shape[0]

        self.futures = {}
        self.executor = ThreadPoolExecutor(max_workers=settings.max_workers_per_dataset)

        worker_nr = 0
        self.nr_background_workers = (
            settings.max_workers_per_dataset - settings.reserved_workers_per_dataset_for_getitem
        )
        idx = 0
        while (
            worker_nr < settings.max_workers_per_dataset - settings.reserved_workers_per_dataset_for_getitem
            and idx < len(self)
        ):
            fut = self.submit(idx)
            if isinstance(fut, Future):
                fut.add_done_callback(self.background_worker_callback)
                idx += 1
                worker_nr += 1
            else:
                idx += 1

        self.stat = DatasetStat(path=data_file_path.with_suffix(".stat.yml"), dataset=self)

    def __len__(self):
        return self._len

    def update_meta(self, meta: dict) -> dict:
        meta = self.dataset.update_meta(meta)
        tensor_meta = meta.get(self.tensor_name, {})
        assert "stat" not in tensor_meta
        tensor_meta["stat"] = self.stat
        meta[self.tensor_name] = tensor_meta
        return meta

    def __getitem__(self, idx) -> typing.OrderedDict[str, numpy.ndarray]:
        idx = int(idx)
        self.submit(idx).result()
        return OrderedDict(**{self.tensor_name: self.data_file[self.tensor_name][idx : idx + 1]})

    def shutdown(self):
        if self.futures:
            for fut in self.futures.values():
                fut.cancel()

        if self.executor is not None:
            self.executor.shutdown()

        self.dataset.shutdown()

    def background_worker_callback(self, fut: Future):
        idx = fut.result()
        for next_idx in range(idx + self.nr_background_workers, len(self), self.nr_background_workers):
            next_fut = self.submit(next_idx)
            if next_fut is not None:
                next_fut.add_done_callback(self.background_worker_callback)
                break

    def ready(self, idx: int) -> bool:
        n5ds = self.data_file[self.tensor_name]
        chunk_idx = tuple([idx] + [0] * (len(n5ds.shape) - 1))
        return n5ds.chunk_exists(chunk_idx)

    def submit(self, idx: int) -> Union[int, Future]:
        if self.ready(idx):
            fut = Future()
            fut.set_result(idx)
        else:
            with N5CachedDataset_submit_lock:
                fut = self.futures.get(idx, None)
                if fut is None:
                    fut = self.executor.submit(self.process, idx)
                    self.futures[idx] = fut

        return fut

    def process(self, idx: int) -> int:
        self.data_file[self.tensor_name][idx, ...] = self.dataset[idx][self.tensor_name]

        return idx


class N5CachedDatasetSubset(torch.utils.data.Subset):
    dataset: N5CachedDataset

    def shutdown(self):
        self.dataset.shutdown()


class ZipDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        datasets: Dict[str, torch.utils.data.Dataset],
        transformation: Callable[[typing.OrderedDict], typing.OrderedDict] = lambda x: x,
    ):
        super().__init__()
        datasets = OrderedDict(**datasets)
        assert len(datasets) > 0
        self._len = len(list(datasets.values())[0])
        assert all(len(ds) == self._len for ds in datasets.values())
        self.datasets = datasets
        self.transformation = transformation

    def __len__(self):
        return self._len

    def get_meta(self, idx: int) -> dict:
        meta = {"idx": idx}
        for name, ds in self.datasets.items():
            if hasattr(ds, "update_meta"):
                meta = ds.update_meta(meta)

        return meta

    def __getitem__(self, idx: int) -> typing.OrderedDict[str, Any]:
        tensors = OrderedDict()
        for name, ds in self.datasets.items():
            tensors[name] = ds[idx][name]

        tensors["meta"] = [self.get_meta(idx)]
        return self.transformation(tensors)

    def shutdown(self):
        for ds in self.datasets:
            if hasattr(ds, "shutdown"):
                ds.shutdown()


class ZipSubset(torch.utils.data.Subset):
    dataset: ZipDataset

    def __init__(self, dataset: ZipDataset, indices: Sequence[int], z_crop: Optional[Tuple[int, int]] = None):
        max_idx = max(indices)
        if z_crop is None:
            assert max_idx < len(dataset), (max_idx, len(dataset))
        else:
            not_cropped_indices = [
                idx
                for idx in range(len(dataset))
                if z_crop[0] <= dataset.get_meta(idx).get("z_slice", z_crop[0]) < z_crop[1]
            ]
            assert max_idx < len(not_cropped_indices), (max_idx, len(not_cropped_indices))
            indices = numpy.asarray(not_cropped_indices)[numpy.asarray(indices)]

        super().__init__(dataset=dataset, indices=indices)

    def shutdown(self):
        self.dataset.shutdown()


def get_collate_fn(batch_transformation: Callable):
    def collate_fn(samples: List[typing.OrderedDict[str, Any]]):
        assert len(samples) > 0
        batch = OrderedDict()
        for b in zip(*[s.items() for s in samples]):
            tensor_names, tensor_batch = zip(*b)
            name = tensor_names[0]
            tensor0 = tensor_batch[0]
            assert all(name == k for k in tensor_names[1:])
            assert all(type(tensor0) is type(v) for v in tensor_batch[1:])
            if isinstance(tensor0, numpy.ndarray):
                tensor_batch = numpy.ascontiguousarray(numpy.concatenate(tensor_batch, axis=0))
            elif isinstance(tensor0, torch.Tensor):
                raise NotImplementedError
                # tensor_batch = torch.cat(tensor_batch, dim=0)
            elif isinstance(tensor0, list):
                assert all(
                    len(v) == 1 for v in tensor_batch
                )  # expect explicit batch dimension! (list of len 1 for all samples)
                tensor_batch = [vv for v in tensor_batch for vv in v]
            else:
                raise NotImplementedError(type(tensor0))

            batch[name] = tensor_batch

        return batch_transformation(batch)

    return collate_fn


class ConcatDataset(torch.utils.data.ConcatDataset):
    def __init__(self, datasets: List[torch.utils.data.ConcatDataset], transform: Optional[Callable] = None):
        self.transform = transform
        super().__init__(datasets=datasets)

    def __getitem__(self, item):
        sample = super().__getitem__(item)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def shutdown(self):
        for ds in self.datasets:
            if hasattr(ds, "shutdown"):
                ds.shutdown()
