from __future__ import annotations

import bisect
import logging
import re
import typing
import warnings
from collections import OrderedDict
from concurrent.futures import Future
from concurrent.futures.thread import ThreadPoolExecutor
from functools import partial
from hashlib import sha224 as hash_algorithm
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import h5py
import imageio
import numpy

import torch.utils.data
import yaml
import z5py

import lnet
import lnet.datasets.filters
from lnet import settings
from lnet.datasets.utils import get_gcamp_z_slice_from_path, get_paths
from lnet.stat import DatasetStat
from lnet.transformations.base import ComposedTransformation

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
        datasets_per_file: int = 1,
        samples_per_dataset: int = 1,
        insert_singleton_axes_at: Sequence[int] = tuple(),
        z_slice: Optional[Union[str, int, Callable[[int], int]]] = None,
        skip_indices: Sequence[int] = tuple(),
        meta: Optional[dict] = None,
        repeat: int = 1,
        tag: Optional[str] = None,
        **kwargs,
    ):
        assert not location.endswith(".h5"), "h5 path to dataset missing .h5/Dataset"
        assert all([len(trf) == 1 for trf in transformations]), [list(trf.keys()) for trf in transformations]
        # data specific asserts
        assert transformations or ".h5" not in location, ".h5 datasets require transformation!"
        if "Heart_tightCrop" in location and z_slice is not None and not isinstance(z_slice, int):
            assert callable(z_slice) and z_slice(0), "z direction is inverted for 'Heart_tightCrop'"

        if z_slice is not None and skip_indices:
            warnings.warn(
                f"z_slice {z_slice}) and skip_indices {skip_indices} specified. skip_indices indices are directly based on path index and ignore z_slice."
            )

        assert isinstance(name, str)
        assert isinstance(root, str)
        assert isinstance(location, str)
        assert isinstance(datasets_per_file, int)
        assert isinstance(samples_per_dataset, int)
        self.name = name
        if tag is None:
            self.tag = name
        else:
            self.tag = tag

        self.root = root
        self.location = location
        self.transformations = list(transformations)
        self.datasets_per_file = datasets_per_file
        self.samples_per_dataset = samples_per_dataset
        self.insert_singleton_axes_at = insert_singleton_axes_at
        self.z_slice = z_slice
        self.skip_indices = skip_indices
        self.meta: dict = meta or {}
        self.repeat = repeat
        self.kwargs = kwargs
        self.path: Path = getattr(settings.data_roots, root) / location

    @property
    def transformations(self) -> List[Dict[str, Any]]:
        return self.__transformations

    @transformations.setter
    def transformations(self, trfs):
        assert isinstance(trfs, list)
        for trf in trfs:
            for kwargs in trf.values():
                assert "apply_to" in kwargs and kwargs["apply_to"] == self.name, (self.name, trf)

        self.__transformations = trfs

    @property
    def description(self):
        return yaml.safe_dump(
            {
                "name": self.name,
                "root": self.root,
                "location": self.location,
                "transformations": [trf for trf in self.transformations if "Assert" not in trf],
                "datasets_per_file": self.datasets_per_file,
                "samples_per_dataset": self.samples_per_dataset,
                "insert_singleton_axes_at": self.insert_singleton_axes_at,
                "z_slice": self.z_slice.__name__ if callable(self.z_slice) else self.z_slice,
                "skip_indices": list(self.skip_indices),
                "kwargs": self.kwargs,
            }
        )


class DatasetFromInfo(torch.utils.data.Dataset):
    get_z_slice: Callable[[int], Optional[int]]
    paths: List[Path]

    def __init__(self, *, info: TensorInfo):
        super().__init__()
        self.tensor_name = info.name
        self.info = info
        self.description = info.description
        self.transform = lnet.transformations.ComposedTransformation(
            *[
                getattr(lnet.transformations, name)(**kwargs)
                for trf in info.transformations
                for name, kwargs in trf.items()
            ]
        )

        self.insert_singleton_axes_at = info.insert_singleton_axes_at

        self._z_slice_mod: Optional[int] = None
        self._z_slice: Optional[int] = None
        self._z_offset = 0
        self._z_slice_from_path: Optional[Callable[[Path], int]] = None
        if info.z_slice is None:
            pass
        elif isinstance(info.z_slice, int):
            self._z_slice = info.z_slice
        elif isinstance(info.z_slice, str):
            if "+" in info.z_slice:
                z_offset, z_slice_str = info.z_slice.split("+")
                self._z_offset = int(z_offset)
            else:
                z_slice_str = info.z_slice

            if z_slice_str.startswith("idx%"):
                self._z_slice_mod = int(z_slice_str[len("idx%") :])
            elif z_slice_str.startswith("from_gcamp_path_"):
                assert info.name == "ls"
                self._z_slice = int(z_slice_str[len("from_gcamp_path_") :])
                retrieved_z_slice = get_gcamp_z_slice_from_path(info.path)
                assert retrieved_z_slice == self._z_slice, (retrieved_z_slice, self._z_slice)
            elif z_slice_str.startswith("from_gcamp_path"):
                self._z_slice_from_path = get_gcamp_z_slice_from_path

            else:
                raise NotImplementedError(info.z_slice)
        else:
            self.get_z_slice = info.z_slice

    def get_z_slice(self, idx: int) -> Optional[int]:
        if self._z_slice_from_path is not None:
            path_idx = idx // self.info.datasets_per_file
            idx %= self.info.datasets_per_file
            return self._z_slice_from_path(self.paths[path_idx])
        elif self._z_slice is None:
            if self._z_slice_mod is None:
                return None
            else:
                return abs(self._z_offset + (idx % self._z_slice_mod))
        else:
            if self._z_slice_mod is None:
                return self._z_offset + self._z_slice
            else:
                raise NotImplementedError("_z_slice and _z_slice_mod?!?")

    def update_meta(self, meta: dict) -> dict:
        tmeta = meta.get(self.tensor_name, {})
        for k, v in self.info.meta.items():
            if k in tmeta:
                assert tmeta[k] == v, (tmeta[k], v)
            else:
                tmeta[k] = v

        has_z_slice = tmeta.get("z_slice", None)
        z_slice = self.get_z_slice(tmeta.get("idx", meta["idx"]))
        if z_slice is not None:
            assert has_z_slice is None or has_z_slice == z_slice
            tmeta["z_slice"] = z_slice
            meta[self.tensor_name] = tmeta

        return meta

    def shutdown(self):
        pass


class TiffDataset(DatasetFromInfo):
    def __init__(self, *, info: TensorInfo):
        if info.samples_per_dataset != 1:
            raise NotImplementedError

        assert not info.kwargs, info.kwargs
        super().__init__(info=info)
        paths = get_paths(info.path)
        self.paths = [p for i, p in enumerate(paths) if i not in info.skip_indices]

    def __len__(self):
        return len(self.paths) * self.info.datasets_per_file * self.info.samples_per_dataset

    def __getitem__(self, idx: int) -> typing.OrderedDict[str, Union[numpy.ndarray, list]]:
        path_idx = idx // self.info.datasets_per_file
        idx %= self.info.datasets_per_file
        img_path = self.paths[path_idx]
        img: numpy.ndarray = imageio.volread(img_path)
        if self.info.datasets_per_file > 1:
            img = img[idx : idx + 1]

        for axis in self.insert_singleton_axes_at:
            img = numpy.expand_dims(img, axis=axis)

        return self.transform(OrderedDict(**{self.tensor_name: img}))


class H5Dataset(DatasetFromInfo):
    @staticmethod
    def get_ds_resolver(pattern: str, results_to: List[str]) -> Callable[[str], None]:
        def ds_resolver(name: str):
            if re.match(pattern, name):
                results_to.append(name)

        return ds_resolver

    def __init__(self, *, info: TensorInfo):
        if info.kwargs:
            raise NotImplementedError(info.kwargs)
        super().__init__(info=info)
        h5_ext = ".h5"
        assert h5_ext in info.path.as_posix(), info.path.as_posix()
        file_path_glob, within_pattern = info.path.as_posix().split(h5_ext)
        within_pattern = within_pattern.strip("/")
        file_path_glob += h5_ext
        paths = get_paths(Path(file_path_glob))
        self.paths = [p for i, p in enumerate(paths) if i not in info.skip_indices]
        self.dataset_paths = []
        for p in paths:
            with h5py.File(p, mode="r") as hf:
                root_group = hf["/"]
                within = []
                root_group.visit(self.get_ds_resolver(within_pattern, within))
                self.dataset_paths.append(sorted(within))

        assert all(self.dataset_paths), self.dataset_paths

    def __len__(self):
        return len(self.paths) * self.info.datasets_per_file * self.info.samples_per_dataset

    def __getitem__(self, idx: int) -> typing.OrderedDict[str, Union[numpy.ndarray, list]]:
        path_idx = idx // (self.info.datasets_per_file * self.info.samples_per_dataset)
        idx %= self.info.datasets_per_file * self.info.samples_per_dataset
        ds_idx = idx // self.info.samples_per_dataset
        idx %= self.info.samples_per_dataset
        dataset_path = self.dataset_paths[path_idx][ds_idx]
        with h5py.File(self.paths[path_idx], mode="r") as hf:
            h5ds = hf[dataset_path]
            if self.info.samples_per_dataset > 1:
                img: numpy.ndarray = h5ds[idx : idx + 1]
            else:
                img: numpy.ndarray = h5ds[:]

        for axis in self.insert_singleton_axes_at:
            img = numpy.expand_dims(img, axis=axis)

        return self.transform(OrderedDict(**{self.tensor_name: img}))


class DatasetFromInfoExtender(torch.utils.data.Dataset):
    def __init__(self, dataset: Union[N5CachedDatasetFromInfo, DatasetFromInfo]):
        assert isinstance(dataset, (DatasetFromInfo, N5CachedDatasetFromInfo)), type(dataset)
        self.dataset = dataset

    def update_meta(self, meta: dict) -> dict:
        return self.dataset.update_meta(meta)

    def shutdown(self):
        self.dataset.shutdown()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


class N5CachedDatasetFromInfo(DatasetFromInfoExtender):
    def __init__(self, dataset: DatasetFromInfo):
        super().__init__(dataset=dataset)
        self.repeat = dataset.info.repeat
        description = dataset.description
        data_file_path = (
            settings.cache_path
            / f"{dataset.info.tag}_{dataset.tensor_name}_{hash_algorithm(description.encode()).hexdigest()}.n5"
        )
        data_file_path.with_suffix(".txt").write_text(description)

        logger.warning("cache %s_%s to %s", dataset.info.tag, dataset.tensor_name, data_file_path)
        tensor_name = self.dataset.tensor_name
        self.data_file = data_file = z5py.File(path=str(data_file_path), mode="a", use_zarr_format=False)
        shape = data_file[tensor_name].shape if tensor_name in data_file else None

        if shape is None:
            _len = len(dataset)
            sample = dataset[0]
            tensor = sample[tensor_name]
            tensor_shape = tuple(tensor.shape)
            assert tensor_shape[0] == 1, tensor_shape  # expected explicit batch dimension
            shape = (_len,) + tensor_shape[1:]
            data_file.create_dataset(tensor_name, shape=shape, chunks=tensor_shape, dtype=tensor.dtype)

        # self.futures = {}
        # global N5CachedDataset_executor, N5CachedDataset_executor_user_count
        # N5CachedDataset_executor_user_count += 1
        # if N5CachedDataset_executor is None:
        #     assert N5CachedDataset_executor_user_count == 1
        #     N5CachedDataset_executor = ThreadPoolExecutor(max_workers=settings.max_workers_per_dataset)

        # worker_nr = 0
        # self.nr_background_workers = (
        #     settings.max_workers_per_dataset - settings.reserved_workers_per_dataset_for_getitem
        # )
        # idx = 0
        # while (
        #     worker_nr < settings.max_workers_per_dataset - settings.reserved_workers_per_dataset_for_getitem
        #     and idx < len(self)
        # ):
        #     fut = self.submit(idx)
        #     if isinstance(fut, Future):
        #         fut.add_done_callback(self.background_worker_callback)
        #         idx += 1
        #         worker_nr += 1
        #     else:
        #         idx += 1

        self.stat = DatasetStat(path=data_file_path.with_suffix(".stat_v1.yml"), dataset=self)

    def update_meta(self, meta: dict) -> dict:
        tensor_meta = meta[self.dataset.tensor_name]
        assert "stat" not in tensor_meta
        tensor_meta["stat"] = self.stat
        meta[self.dataset.tensor_name] = tensor_meta
        return super().update_meta(meta)

    def __getitem__(self, idx) -> typing.OrderedDict[str, numpy.ndarray]:
        idx = int(idx)
        idx //= self.repeat
        self.submit(idx)  # .result()
        z5dataset = self.data_file[self.dataset.tensor_name]
        assert idx < z5dataset.shape[0], z5dataset.shape
        return OrderedDict([(self.dataset.tensor_name, z5dataset[idx : idx + 1])])

    def __len__(self):
        return len(self.dataset) * self.repeat

    def shutdown(self):
        # if self.futures:
        #     for fut in self.futures.values():
        #         fut.cancel()

        # global N5CachedDataset_executor, N5CachedDataset_executor_user_count
        # N5CachedDataset_executor_user_count -= 1
        # if N5CachedDataset_executor_user_count == 0:
        #     N5CachedDataset_executor.shutdown()
        #     N5CachedDataset_executor = None

        super().shutdown()

    # def background_worker_callback(self, fut: Future):
    #     idx = fut.result()
    #     for next_idx in range(idx + self.nr_background_workers, len(self), self.nr_background_workers):
    #         next_fut = self.submit(next_idx)
    #         if next_fut is not None:
    #             next_fut.add_done_callback(self.background_worker_callback)
    #             break

    def ready(self, idx: int) -> bool:
        n5ds = self.data_file[self.dataset.tensor_name]
        chunk_idx = tuple([idx] + [0] * (len(n5ds.shape) - 1))
        return n5ds.chunk_exists(chunk_idx)

    def submit(self, idx: int) -> Union[int, Future]:
        if self.ready(idx):
            # fut = Future()
            # fut.set_result(idx)
            return idx
        else:
            assert self.repeat == 1, "could write same idx in parallel"
            return self.process(idx)
            # with N5CachedDataset_submit_lock:
            # fut = self.futures.get(idx, None)
            # if fut is None:
            #     fut = N5CachedDataset_executor.submit(self.process, idx)
            #     self.futures[idx] = fut

        # return fut

    def process(self, idx: int) -> int:
        self.data_file[self.dataset.tensor_name][idx, ...] = self.dataset[idx][self.dataset.tensor_name]

        return idx


class N5CachedDatasetFromInfoSubset(DatasetFromInfoExtender):
    dataset: N5CachedDatasetFromInfo

    def __init__(
        self,
        dataset: N5CachedDatasetFromInfo,
        indices: Optional[Sequence[int]] = None,
        filters: Sequence[Tuple[str, Dict[str, Any]]] = tuple(),
    ):
        super().__init__(dataset=dataset)
        assert isinstance(dataset, N5CachedDatasetFromInfo)
        description = (
            dataset.dataset.description
            + "\n"
            + yaml.safe_dump(
                {"indices": None if indices is None else list(indices), "filters": [list(fil) for fil in filters]}
            )
        )
        self.description = description
        indices = numpy.arange(len(dataset)) if indices is None else indices
        mask_file_path = settings.cache_path / f"{hash_algorithm(description.encode()).hexdigest()}.index_mask.npy"
        mask_description_file_path = mask_file_path.with_suffix(".txt")
        if not mask_description_file_path.exists():
            mask_description_file_path.write_text(description)

        logger.info("using dataset mask %s", mask_description_file_path)
        if mask_file_path.exists():
            mask: numpy.ndarray = numpy.repeat(numpy.load(str(mask_file_path)), dataset.repeat)
        else:
            assert dataset.repeat == 1, "don't compute anything on the repeated dataset!"
            mask = numpy.zeros(len(dataset), dtype=bool)
            mask[numpy.asarray(indices)] = True
            for name, kwargs in filters:
                for k, v in kwargs.items():
                    if isinstance(v, dict) and len(v) == 1:
                        kk = list(v.keys())[0]
                        if kk == "percentile":
                            kwargs[k] = dataset.stat.get_percentiles(dataset.dataset.tensor_name, [v[kk]])
                        elif kk == "mean+xstd":
                            mean, std = dataset.stat.get_mean_std(dataset.dataset.tensor_name, (5.0, 99.9))
                            kwargs[k] = mean + std * v[kk]
                        else:
                            raise NotImplementedError

            filters = [
                partial(getattr(lnet.datasets.filters, name), dataset=dataset, **kwargs) for name, kwargs in filters
            ]

            def apply_filters_to_mask(idx: int) -> None:
                if any([not fil(idx=idx) for fil in filters]):
                    mask[idx] = False

            with ThreadPoolExecutor(max_workers=settings.max_workers_per_dataset) as executor:
                futs = []
                for idx in indices:
                    futs.append(executor.submit(apply_filters_to_mask, idx))

                for idx, fut in enumerate(futs):
                    exc = fut.exception()
                    if exc is not None:
                        raise exc

            numpy.save(str(mask_file_path), mask)

        self.mask = mask

    def shutdown(self):
        self.dataset.shutdown()

    @property
    def mask(self):
        return self.__mask

    @mask.setter
    def mask(self, new_mask):
        self.indices = numpy.arange(len(self.dataset))[new_mask]
        self.__mask = new_mask

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def update_meta(self, meta: dict) -> dict:
        common_idx = meta.get("idx", None)
        if common_idx is not None:
            idx = self.indices[common_idx]
            tmeta = meta.get(self.dataset.dataset.tensor_name, {})
            idx_is = tmeta.get("idx", None)
            if idx_is is None:
                tmeta["idx"] = idx
                meta[self.dataset.dataset.tensor_name] = tmeta
            else:
                assert idx_is == idx, f"expected existing idx to be equal to {idx} or none, but got {idx_is}"

        return super().update_meta(meta)


def get_dataset_from_info(
    info: TensorInfo,
    *,
    transformations: Sequence[Dict[str, dict]] = tuple(),
    cache: bool = False,
    indices: Optional[Sequence[int]] = None,
    filters: Sequence[Tuple[str, Dict[str, Any]]] = tuple(),
) -> Union[DatasetFromInfo, N5CachedDatasetFromInfo, N5CachedDatasetFromInfoSubset]:
    info.transformations += list(transformations)
    if info.location.endswith(".tif"):
        ds = TiffDataset(info=info)
    elif ".h5" in info.location:
        ds = H5Dataset(info=info)
    else:
        raise NotImplementedError(info.location)

    if (indices or filters) and not cache:
        raise NotImplementedError("subset only implemented for cached dataset")

    if cache:
        ds = N5CachedDatasetFromInfo(dataset=ds)
        ds = N5CachedDatasetFromInfoSubset(dataset=ds, indices=indices, filters=filters)

    return ds


class ZipDataset(torch.utils.data.Dataset):
    """Zip N5CachedDatasetSubsets and adapt them by joining indice masks """

    def __init__(
        self,
        datasets: typing.OrderedDict[str, Union[N5CachedDatasetFromInfoSubset, DatasetFromInfo]],
        transformation: Callable[[typing.OrderedDict], typing.OrderedDict] = lambda x: x,
        join_dataset_masks: bool = True,
    ):
        super().__init__()
        datasets: typing.OrderedDict[str, N5CachedDatasetFromInfoSubset] = OrderedDict(**datasets)
        assert len(datasets) > 0

        if join_dataset_masks:
            base_len = len(list(datasets.values())[0].dataset)
            assert all([len(ds.dataset) == base_len for ds in datasets.values()]), (
                [len(ds.dataset) for ds in datasets.values()],
                [ds.description for ds in datasets.values()],
            )
            assert all(
                [isinstance(ds, N5CachedDatasetFromInfoSubset) for ds in datasets.values()]
            ), f"can only join N5CachedDatasetFromInfoSubset, {[type(ds) for ds in datasets.values()]}"
            joined_mask = numpy.logical_and.reduce(numpy.stack([ds.mask for ds in datasets.values()]))
            for ds in datasets.values():
                ds.mask = joined_mask

        _len = len(list(datasets.values())[0])
        assert all(len(ds) == _len for ds in datasets.values())
        self._len = _len
        self.datasets = datasets
        self.transformation = transformation

    def __len__(self):
        return self._len

    def get_meta(self, idx: int) -> dict:
        meta = {"idx": idx}
        for name, ds in self.datasets.items():
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
                tensor_batch = torch.cat(tensor_batch, dim=0)
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

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        for tmeta in sample["meta"]:
            tmeta["dataset_idx"] = tmeta.get("dataset_idx", []) + [dataset_idx]

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def shutdown(self):
        for ds in self.datasets:
            if hasattr(ds, "shutdown"):
                ds.shutdown()
