from __future__ import annotations

import bisect
import logging
import re
import warnings
from concurrent.futures.thread import ThreadPoolExecutor
from functools import partial
from hashlib import sha224 as hash_algorithm
from pathlib import Path
from time import sleep
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import h5py
import imageio
import numpy
import torch.utils.data
import yaml
import z5py

import hylfm
import hylfm.datasets.filters
from hylfm import settings
from hylfm.datasets.utils import get_paths, merge_nested_dicts
from hylfm.hylfm_types import TransformLike
from hylfm.stat_ import DatasetStat

logger = logging.getLogger(__name__)


class PathOfInterest:
    def __init__(self, *points: Tuple[int, int, int, int], sigma: int = 1):
        self.points = points
        self.sigma = sigma


class TensorInfo:
    def __init__(
        self,
        *,
        name: str,
        root: Union[str, Path] = Path(),
        location: str,
        transforms: Sequence[Dict[str, Any]] = tuple(),
        datasets_per_file: int = 1,
        samples_per_dataset: int = 1,
        remove_singleton_axes_at: Sequence[int] = tuple(),
        insert_singleton_axes_at: Sequence[int] = tuple(),
        z_slice: Optional[Union[str, int, Callable[[int], int]]] = None,
        skip_indices: Sequence[int] = tuple(),
        meta: Optional[dict] = None,
        repeat: int = 1,
        tag: Optional[str] = None,
        **kwargs,
    ):
        assert not location.endswith(".h5"), "h5 path to dataset missing .h5/Dataset"
        assert all([len(trf) == 1 for trf in transforms]), [list(trf.keys()) for trf in transforms]
        # data specific asserts
        if "Heart_tightCrop" in location and z_slice is not None and not isinstance(z_slice, int):
            assert callable(z_slice) and z_slice(0), "z direction is inverted for 'Heart_tightCrop'"

        if z_slice is not None and skip_indices:
            warnings.warn(
                f"z_slice {z_slice}) and skip_indices {skip_indices} specified. "
                f"skip_indices indices are directly based on path index and ignore z_slice."
            )

        assert isinstance(name, str)
        assert isinstance(root, (str, Path))
        assert isinstance(location, str)
        assert isinstance(datasets_per_file, int)
        assert isinstance(samples_per_dataset, int), samples_per_dataset
        self.name = name
        if tag is None:
            self.tag = name
        else:
            self.tag = tag

        self.location = location
        self.transforms = list(transforms)
        self.datasets_per_file = datasets_per_file
        self.samples_per_dataset = samples_per_dataset
        self.remove_singleton_axes_at = remove_singleton_axes_at
        self.insert_singleton_axes_at = insert_singleton_axes_at
        self.z_slice = z_slice
        self.skip_indices = skip_indices
        self.meta: dict = meta or {}
        self.repeat = repeat
        self.kwargs = kwargs
        self.path: Path = (settings.data_roots[root] if isinstance(root, str) else root) / location
        self.root = root

    @property
    def transforms(self) -> List[Dict[str, Any]]:
        return self.__transforms

    @transforms.setter
    def transforms(self, trfs):
        assert isinstance(trfs, list)
        for trf in trfs:
            for kwargs in trf.values():
                if "apply_to" not in kwargs:
                    raise ValueError(f"missing `apply_to` arg in transformation {trf}")

                if kwargs["apply_to"] != self.name:
                    raise ValueError(f"`TensorInfo.name` {self.name} does not match transformation's `apply_to` {trf}")

        self.__transforms = trfs

    @property
    def description(self):
        descr = {
            "root": str(self.root),
            "location": self.location,
            "transformations": [trf for trf in self.transforms if "Assert" not in trf],
            "datasets_per_file": self.datasets_per_file,
            "samples_per_dataset": self.samples_per_dataset,
            "remove_singleton_axes_at": self.remove_singleton_axes_at,
            "insert_singleton_axes_at": self.insert_singleton_axes_at,
            "z_slice": self.z_slice.__name__ if callable(self.z_slice) else self.z_slice,
            "skip_indices": list(self.skip_indices),
            "kwargs": self.kwargs,
            "repeat": self.repeat,
        }

        return yaml.safe_dump(descr)


class DatasetFromInfo(torch.utils.data.Dataset):
    get_z_slice: Callable[[int], Optional[int]]
    paths: List[Path]

    def __init__(self, *, info: TensorInfo):
        super().__init__()
        self.tensor_name = info.name
        self.info = info
        self.description = info.description
        from hylfm.transforms import ComposedTransform

        self.transform = ComposedTransform(
            *[getattr(hylfm.transforms, name)(**kwargs) for trf in info.transforms for name, kwargs in trf.items()]
        )

        self.remove_singleton_axes_at = info.remove_singleton_axes_at
        self.insert_singleton_axes_at = info.insert_singleton_axes_at

        self._z_slice_mod: Optional[int] = None
        self._z_slice: Optional[int] = None
        self._z_offset = 0
        self._z_step: int = 1
        # self._z_slice_from_path: Optional[Callable[[Path], int]] = None
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
                if "*" in z_slice_str:
                    mod_str, step_str = z_slice_str[len("idx%") :].split("*")
                    self._z_slice_mod = int(mod_str)
                    self._z_step = int(step_str)
                else:
                    self._z_slice_mod = int(z_slice_str[len("idx%") :])
            else:
                raise NotImplementedError(info.z_slice)
        else:
            self.get_z_slice = info.z_slice

    def __getitem__(self, idx: int):
        return {"z_slice": self.get_z_slice(idx)}

    def get_z_slice(self, idx: int) -> Optional[int]:
        if self._z_slice is None:
            if self._z_slice_mod is None:
                return None
            else:
                return abs(self._z_offset + (idx % self._z_slice_mod) * self._z_step)
        else:
            if self._z_slice_mod is None:
                return self._z_offset + self._z_slice
            else:
                raise NotImplementedError("_z_slice and _z_slice_mod?!?")


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

    def __getitem__(self, idx: int) -> Dict[str, Union[numpy.ndarray, list]]:
        sample = super().__getitem__(idx)
        path_idx = idx // self.info.datasets_per_file
        idx %= self.info.datasets_per_file
        img_path = self.paths[path_idx]

        try:
            img: numpy.ndarray = imageio.volread(img_path)
        except Exception as e:
            logger.error("Cannot load %s due to %s", img_path, e)
            raise e

        img = numpy.asarray(img)
        if self.info.datasets_per_file > 1:
            img = img[idx : idx + 1]

        for axis in self.remove_singleton_axes_at:
            if img.shape[axis] == 1:
                img = numpy.squeeze(img, axis=axis)

        for axis in self.insert_singleton_axes_at:
            img = numpy.expand_dims(img, axis=axis)

        while len(img.shape) > 5 and img.shape[0] == 1:
            logger.warning("squeeze(0) %s", img.shape)
            img = img.squeeze(0)

        # assert len(img.shape) == 5, (self.info.name, idx, img.shape)

        sample[self.tensor_name] = img
        sample["batch_len"] = img.shape[0]
        return self.transform(sample)


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

        if info.remove_singleton_axes_at:
            raise NotImplementedError

        super().__init__(info=info)
        self._paths = None
        self._dataset_paths = None
        h5_ext = ".h5"
        assert h5_ext in self.info.path.as_posix(), self.info.path.as_posix()
        file_path_glob, within_pattern = self.info.path.as_posix().split(h5_ext)
        self.file_path_glob = Path(file_path_glob + h5_ext)
        self.within_pattern = within_pattern.strip("/")

    @property
    def paths(self):
        if self._paths is None:
            paths = get_paths(self.file_path_glob)
            self._paths = [p for i, p in enumerate(paths) if i not in self.info.skip_indices]

        return self._paths

    @property
    def dataset_paths(self):
        if self._dataset_paths is None:
            self._dataset_paths = []
            for p in self.paths:
                with h5py.File(p, mode="r") as hf:
                    root_group = hf["/"]
                    within = []
                    root_group.visit(self.get_ds_resolver(self.within_pattern, within))
                    self._dataset_paths.append(sorted(within))

            assert all(self._dataset_paths), self._dataset_paths

        return self._dataset_paths

    def __len__(self):
        return len(self.paths) * self.info.datasets_per_file * self.info.samples_per_dataset

    def __getitem__(self, idx: int) -> Dict[str, Union[numpy.ndarray, list]]:
        sample = super().__getitem__(idx)
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

        sample[self.tensor_name] = img
        sample["batch_len"] = img.shape[0]
        return self.transform(sample)


class DatasetFromInfoExtender(torch.utils.data.Dataset):
    def __init__(self, dataset: Union[N5CachedDatasetFromInfo, DatasetFromInfo]):
        assert isinstance(dataset, (DatasetFromInfo, N5CachedDatasetFromInfo)), type(dataset)
        self.dataset = dataset

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
            settings.cache_dir
            / f"{dataset.info.tag}_{dataset.tensor_name}_{hash_algorithm(description.encode()).hexdigest()}.n5"
        )
        data_file_path.with_suffix(".txt").write_text(description)

        self.from_source = not dataset.transform.transforms
        if not self.from_source:
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

        self.stat = None
        self.stat = DatasetStat(path=data_file_path.with_suffix(".stat_v1.yml"), dataset=self)

    def __getitem__(self, idx) -> Dict[str, Union[List[Dict[str, DatasetStat]], numpy.ndarray]]:
        idx = int(idx)
        phys_idx = idx // self.repeat
        if self.from_source:
            tensor = self.dataset[phys_idx][self.dataset.tensor_name]
        else:
            if idx % self.repeat == 0:
                self.submit(phys_idx)  # .result()
            else:
                for patience in range(50):
                    if self.ready(phys_idx):
                        break
                    else:
                        sleep(0.5)
                else:
                    raise RuntimeError(f"idx {phys_idx} not ready, but requested as idx//repeat {idx}//{self.repeat}")

            z5dataset = self.data_file[self.dataset.tensor_name]
            assert phys_idx < z5dataset.shape[0], z5dataset.shape
            tensor = z5dataset[phys_idx : phys_idx + 1]

        batch_len = tensor.shape[0]
        mini_batch = {
            "batch_len": batch_len,
            self.dataset.tensor_name: tensor,
            "stat": [{self.dataset.tensor_name: self.stat}] * batch_len,
            **{
                k: v if k == "crop_name" else [v] * batch_len for k, v in self.dataset.info.meta.items()
            },  # crop_name is a shared key across any mini-batch
        }
        z_slice = self.dataset.get_z_slice(phys_idx)
        if z_slice is not None:
            mini_batch["z_slice"] = [z_slice] * batch_len

        return mini_batch

    def __len__(self):
        return len(self.dataset) * self.repeat

    def ready(self, idx: int) -> bool:
        n5ds = self.data_file[self.dataset.tensor_name]
        chunk_idx = tuple([idx] + [0] * (len(n5ds.shape) - 1))
        return n5ds.chunk_exists(chunk_idx)

    def submit(self, idx: int) -> int:
        if self.ready(idx):
            return idx
        else:
            return self.process(idx)

    def process(self, idx: int) -> int:
        self.data_file[self.dataset.tensor_name][idx, ...] = self.dataset[idx][self.dataset.tensor_name]

        return idx


class N5CachedDatasetFromInfoSubset(DatasetFromInfoExtender):
    dataset: N5CachedDatasetFromInfo

    def __init__(
        self,
        dataset: N5CachedDatasetFromInfo,
        indices: Optional[Union[slice, Sequence[int]]] = None,
        filters: Sequence[Tuple[str, Dict[str, Any]]] = tuple(),
    ):
        super().__init__(dataset=dataset)
        assert isinstance(dataset, N5CachedDatasetFromInfo)
        description = (
            dataset.dataset.description
            + "\n"
            + yaml.safe_dump(
                {
                    "indices": None
                    if indices is None
                    else str(indices)
                    if isinstance(indices, slice)
                    else list(indices),
                    "filters": [list(fil) for fil in filters],
                }
            )
        )
        self.description = description
        indices = slice(None) if indices is None else indices
        indices = numpy.arange(len(dataset))[indices] if isinstance(indices, slice) else indices
        mask_file_path = (
            settings.cache_dir
            / f"{dataset.dataset.info.tag}_{hash_algorithm(description.encode()).hexdigest()}.index_mask.npy"
        )
        mask_description_file_path = mask_file_path.with_suffix(".txt")

        mask_description_file_path.write_text(description)

        logger.warning("using dataset mask %s", mask_description_file_path)
        if mask_file_path.exists():
            # mask: numpy.ndarray = numpy.repeat(numpy.load(str(mask_file_path)), dataset.repeat)
            mask: numpy.ndarray = numpy.load(str(mask_file_path))
        else:
            if dataset.repeat > 1:
                warnings.warn("computing stat on a repeated dataset!")

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

            filters = [
                partial(getattr(hylfm.datasets.filters, name), dataset=dataset, **kwargs) for name, kwargs in filters
            ]

            def apply_filters_to_mask(i: int) -> None:
                if any([not fil(idx=i) for fil in filters]):
                    mask[i] = False

            if settings.max_workers_per_dataset:
                with ThreadPoolExecutor(max_workers=settings.max_workers_per_dataset) as executor:
                    futs = []
                    for idx in indices:
                        futs.append(executor.submit(apply_filters_to_mask, idx))

                    for idx, fut in enumerate(futs):
                        exc = fut.exception()
                        if exc is not None:
                            raise exc
            else:
                for idx in indices:
                    apply_filters_to_mask(idx)

            numpy.save(str(mask_file_path), mask)

        self.mask = mask

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
        ret = self.dataset[self.indices[idx]]
        ret["idx"] = [idx]
        return ret


def get_dataset_from_info(
    info: TensorInfo,
    *,
    transforms: Sequence[Dict[str, dict]] = tuple(),
    cache: bool = False,
    indices: Optional[Union[slice, Sequence[int]]] = None,
    filters: Sequence[Tuple[str, Dict[str, Any]]] = tuple(),
) -> Union[DatasetFromInfo, N5CachedDatasetFromInfo, N5CachedDatasetFromInfoSubset]:
    info.transforms += list(transforms)
    if info.location.endswith(".tif"):
        ds = TiffDataset(info=info)
    elif ".h5/" in info.path.as_posix():
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
        datasets: Dict[str, Union[N5CachedDatasetFromInfoSubset, DatasetFromInfo]],
        transform: Optional[TransformLike] = None,
        join_dataset_masks: bool = True,
    ):
        super().__init__()
        datasets: Dict[str, N5CachedDatasetFromInfoSubset] = datasets
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
        assert all(len(ds) == _len for ds in datasets.values()), {name: len(ds) for name, ds in datasets.items()}
        self._len = _len
        self.datasets = datasets
        self.transform = transform

    def __len__(self):
        return self._len

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = {}
        for ds in self.datasets.values():
            sample = merge_nested_dicts(sample, ds[idx])

        if self.transform is not None:
            sample = self.transform(sample)

        sample["idx"] = [idx]
        return sample


class ConcatDataset(torch.utils.data.ConcatDataset):
    def __init__(self, datasets: List[torch.utils.data.ConcatDataset], transform: Optional[TransformLike] = None):
        self.transform = transform
        super().__init__(datasets=datasets)

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        sample["dataset_idx"] = [dataset_idx]

        if self.transform is not None:
            sample = self.transform(sample)

        return sample
