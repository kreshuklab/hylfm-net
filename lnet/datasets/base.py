import logging
import re
import threading
import typing
from collections import OrderedDict
from concurrent.futures import Future
from concurrent.futures.thread import ThreadPoolExecutor
from hashlib import sha224 as hash_algorithm
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type, Union

import h5py
import numpy
import torch.utils.data
import yaml
import z5py
from imageio import imread
from scipy.ndimage import zoom

import lnet
from lnet import settings
from lnet.datasets.utils import get_paths_and_numbers
from lnet.registration import (
    BDVTransform,
    Heart_tightCrop_Transform,
    fast_cropped_6ms_Transform,
    fast_cropped_8ms_Transform,
    staticHeartFOV_Transform,
    wholeFOV_Transform,
)
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
        name: str,
        root: str,
        location: str,
        transforms: List[Dict[str, Any]],
        meta: Optional[dict] = None,
        **kwargs,
    ):
        assert isinstance(name, str)
        self.name = name
        self.transforms = transforms
        self.meta: dict = meta or {}
        self.kwargs = kwargs
        self.description = yaml.safe_dump(
            {"name": name, "root": root, "location": location, "transformations": transforms, "meta": meta, "kwargs": kwargs}
        )
        self.location: Path = getattr(settings.data_roots, root) / location


class NamedDatasetInfo:
    x_path: Path
    y_path: Path
    paths: List[Path]
    x_roi: Tuple[slice, slice, slice]
    y_roi: Tuple[slice, slice, slice, slice]
    rois: List[Tuple[slice, ...]]
    # stat: Optional[DatasetStat]
    interesting_paths: Optional[List[PathOfInterest]]

    description: str = ""
    common_path: Path = Path("/")

    def __init__(
        self,
        path: Union[str, Path],
        x_dir: str,
        y_dir: Optional[str] = None,
        description="",
        x_roi: Optional[Tuple[slice, slice]] = None,
        y_roi: Optional[Tuple[slice, slice, slice]] = None,
        # stat: Optional[DatasetStat] = None,
        interesting_paths: Optional[List[PathOfInterest]] = None,
        length: Optional[int] = None,
        x_shape: Optional[Tuple[int, int]] = None,
        y_shape: Optional[Tuple[int, int, int]] = None,
        AffineTransform: Optional[Union[str, Type[BDVTransform]]] = None,
        z_slices: Optional[Sequence[int]] = None,
        dynamic_z_slice_mod: Optional[int] = None,
    ):
        if z_slices is not None:
            assert AffineTransform is not None

        self.x_path = self.common_path / path / x_dir
        self.y_path = None if y_dir is None else self.common_path / path / y_dir

        self.description = description or self.description

        if isinstance(AffineTransform, str):
            if AffineTransform == "from_x_path":
                posix_path = self.x_path.as_posix()
                indicators_and_AffineTransforms = {
                    "fast_cropped_6ms": fast_cropped_6ms_Transform,
                    "fast_cropped_8ms": fast_cropped_8ms_Transform,
                    "Heart_tightCrop": Heart_tightCrop_Transform,
                    "staticHeartFOV": staticHeartFOV_Transform,
                    "wholeFOV": wholeFOV_Transform,
                }
                for tag, TransformClass in indicators_and_AffineTransforms.items():
                    if tag in posix_path:
                        assert AffineTransform == "from_x_path"  # make sure tag is found only once
                        AffineTransform = TransformClass

            else:
                raise NotImplementedError(AffineTransform)

        self.DefaultAffineTransform = AffineTransform
        if AffineTransform is not None:
            x_shape = x_shape or AffineTransform.lf_shape[1:]

            auto_y_shape = tuple(
                y - y_crop[0] - y_crop[1] for y_crop, y in zip(AffineTransform.lf2ls_crop, AffineTransform.ls_shape)
            )
            auto_y_roi = tuple(
                slice(y_crop[0], y - y_crop[1])
                for y_crop, y in zip(AffineTransform.lf2ls_crop, AffineTransform.ls_shape)
            )
            if z_slices is not None or dynamic_z_slice_mod is not None:
                # auto_y_shape = auto_y_shape[1:]
                auto_y_roi = auto_y_roi[1:]

            y_shape = y_shape or auto_y_shape
            y_roi = y_roi or auto_y_roi

            if z_slices is not None or dynamic_z_slice_mod is not None:
                assert len(y_shape) == 3
                assert len(y_roi) == 2

        self.x_roi = (slice(None), slice(None), slice(None)) if x_roi is None else (slice(None),) + x_roi
        self.y_roi = (slice(None), slice(None), slice(None), slice(None)) if y_roi is None else (slice(None),) + y_roi

        self.interesting_paths = interesting_paths
        self.length = length

        self.x_shape = x_shape
        self.y_shape = y_shape

        self.z_slices = z_slices
        self.dynamic_z_slice_mod = dynamic_z_slice_mod

        self.paths = [self.x_path]
        self.rois = [self.x_roi]
        self.shapes = [self.x_shape]
        if self.y_path is not None:
            self.paths.append(self.y_path)
            self.rois.append(self.y_roi)
            self.shapes.append(self.y_shape)


def resize(
    arr: numpy.ndarray, output_shape: Tuple[int, ...], roi: Optional[Tuple[slice, ...]] = None, order: int = 1
) -> numpy.ndarray:
    assert 0 <= order <= 5, order
    if roi is not None:
        assert len(arr.shape) == len(roi)
        arr = arr[roi]

    assert len(arr.shape) == len(output_shape), (arr.shape, output_shape)
    assert all([sin >= sout for sin, sout in zip(arr.shape, output_shape)]), (arr.shape, output_shape)

    if arr.shape == output_shape:
        return arr
    else:
        return zoom(arr, [sout / sin for sin, sout in zip(arr.shape, output_shape)], order=order)


class DatasetFromInfo(torch.utils.data.Dataset):
    def __init__(self, *, info: TensorInfo):
        super().__init__()
        self.tensor_name = info.name
        self.description = info.description
        self.transform = lnet.transformations.ComposedTransform(
            *[getattr(lnet.transformations, name)(**kwargs) for trf in info.transforms for name, kwargs in trf.items()]
        )

    def update_meta(self, meta: dict) -> dict:
        return meta

    def shutdown(self):
        pass


class TiffDataset(DatasetFromInfo):
    def __init__(self, *, info: TensorInfo):
        given_kwargs = dict(info.kwargs)
        info.kwargs = {
            "in_batches_of": given_kwargs.pop("in_batches_of", 1),
            "insert_singleton_axes_at": given_kwargs.pop("insert_singleton_axes_at", []),
        }
        assert not given_kwargs, given_kwargs
        super().__init__(info=info)
        paths, numbers = get_paths_and_numbers(info.location)
        self.paths = paths
        self.numbers = numbers

        self.in_batches_of: int = info.kwargs["in_batches_of"]
        self.insert_singleton_axes_at: List[int] = info.kwargs["insert_singleton_axes_at"]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int) -> typing.OrderedDict[str, Union[numpy.ndarray, list]]:
        path_idx = idx // self.in_batches_of
        idx %= self.in_batches_of
        img_path = self.paths[path_idx]
        img: numpy.ndarray = imread(img_path)
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
        given_kwargs = dict(info.kwargs)
        info.kwargs = {
            "in_batches_of": given_kwargs.pop("in_batches_of", 1),
            "insert_singleton_axes_at": given_kwargs.pop("insert_singleton_axes_at", []),
        }
        assert not given_kwargs, given_kwargs
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
        self.in_batches_of: int = info.kwargs["in_batches_of"]
        self.insert_singleton_axes_at: List[int] = info.kwargs["insert_singleton_axes_at"]
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
        print("h5 shutdown!")
        self._shutdown = True
        [hf.close() for hf in self.h5files]


def get_dataset_from_info(info: TensorInfo) -> DatasetFromInfo:
    if str(info.location).endswith(".tif"):
        return TiffDataset(info=info)
    elif ".h5" in str(info.location):
        return H5Dataset(info=info)
    else:
        raise NotImplementedError


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

        self.submit_lock = threading.Lock()
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
        with self.submit_lock:
            if self.ready(idx):
                fut = Future()
                fut.set_result(idx)
            else:
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
        transform: Callable[[typing.OrderedDict], typing.OrderedDict] = lambda x: x,
    ):
        super().__init__()
        datasets = OrderedDict(**datasets)
        assert len(datasets) > 0
        self._len = len(list(datasets.values())[0])
        assert all(len(ds) == self._len for ds in datasets.values())
        self.datasets = datasets
        self.transform = transform

    def __len__(self):
        return self._len

    def __getitem__(self, idx: int) -> typing.OrderedDict[str, Any]:
        meta = {"idx": idx}
        tensors = OrderedDict()
        for name, ds in self.datasets.items():
            if hasattr(ds, "update_meta"):
                meta = ds.update_meta(meta)
            tensors[name] = ds[idx][name]

        tensors["meta"] = [meta]
        return self.transform(tensors)

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
