import logging
import numpy
import os
import re
import torch.utils.data
import z5py

from concurrent.futures import Future, as_completed
from concurrent.futures.thread import ThreadPoolExecutor
from hashlib import sha224 as hash
from inferno.io.transform import Transform, Compose
from pathlib import Path
from scipy.interpolate import griddata
from scipy.ndimage import zoom
from tifffile import imread, imsave
from typing import List, Optional, Tuple, Union, Callable, Sequence, Generator, Dict, Any

from lnet.dataset_configs import PathOfInterest, DatasetConfigEntry
from lnet.stat import DatasetStat

logger = logging.getLogger(__name__)


def resize(
    arr: numpy.ndarray, output_shape: Tuple[int, ...], roi: Optional[Tuple[slice, ...]] = None, order: int = 1
) -> numpy.ndarray:
    assert 0 <= order <= 5, order
    if roi is not None:
        arr = arr[roi]

    arr_shape = arr.shape
    if len(arr_shape) == len(output_shape):
        pass
    elif len(arr_shape) + 1 == len(output_shape):
        assert output_shape[0] == 1, (arr_shape, output_shape)
        arr_shape = (1,) + arr_shape
        arr = arr[None,]
    else:
        raise ValueError(f"{arr_shape} {output_shape}")

    assert len(arr_shape) == len(output_shape), (arr_shape, output_shape)
    assert all([sout <= sin for sin, sout in zip(arr_shape, output_shape)]), (arr_shape, output_shape)

    return zoom(arr, [sout / sin for sin, sout in zip(arr_shape, output_shape)], order=order)


class N5Dataset(torch.utils.data.Dataset):
    x_paths: List[Path]
    y_paths: List[Path]
    data_file: Optional[z5py.File] = None
    futures: Optional[List[Tuple[Future, Future]]] = None

    def __init__(
        self,
        name: str,
        x_folder: Path,
        y_folder: Path,
        x_shape: Tuple[int, int],
        y_shape: Tuple[int, int, int],
        original_y_shape: Tuple[int, int, int],
        interesting_paths: Optional[List[PathOfInterest]] = None,
        x_roi: Tuple[slice, slice] = (slice(None), slice(None)),
        y_roi: Tuple[slice, slice, slice] = (slice(None), slice(None), slice(None)),
        interpolation_order: int = 3,
        n: Optional[int] = None,
        n_indices: Optional[List[int]] = None,
        save: bool = True,
        transforms: Optional[List[Union[Transform, Callable[[DatasetStat], Generator[Transform, None, None]]]]] = None,
        has_aux: bool = False,
        data_folder: Optional[Path] = None,
    ):
        super().__init__()
        x_shape = (1,) + x_shape
        y_shape = (1,) + y_shape

        if data_folder is None:
            data_folder = os.environ.get("DATA_FOLDER", None)
            if data_folder is None:
                data_folder =  Path(__file__).parent.parent / "data"
            else:
                data_folder = Path(data_folder)

        assert data_folder.exists(), data_folder.absolute()
        assert n is None or n_indices is None, "Cannot select n and n_indices at once"
        if n is not None:
            n_indices = list(numpy.arange(n))
        elif n_indices is None:
            n_indices = slice(None)

        assert x_folder.exists(), x_folder
        assert y_folder.exists(), y_folder
        self.x_shape = x_shape
        self.y_shape = y_shape
        self.interpolation_order = interpolation_order
        shapestr = f"n: {n}\ninterpolation order: {interpolation_order}\nx roi: {x_roi}\ny roi: {y_roi}\nx shape: {x_shape}\ny shape: {y_shape}"
        data_file_name = (
            data_folder
            / f"{name}_{hash((shapestr + x_folder.as_posix() + y_folder.as_posix()).encode()).hexdigest()}.n5"
        )
        with Path(data_file_name.as_posix().replace(".n5", ".txt")).open("w") as f:
            f.write(shapestr)

        logger.info("data_file_name %s", data_file_name)
        if data_file_name.exists():
            self.data_file = z5py.File(path=data_file_name.as_posix(), mode="r", use_zarr_format=False)
            self._len = self.data_file["x"].shape[0]
        else:
            raw_x_paths = list(map(str, Path(x_folder).glob("*.tif")))
            raw_y_paths = list(map(str, Path(y_folder).glob("*.tif")))
            x_numbers = [tuple(int(nr) for nr in re.findall(r"\d+", os.path.basename(p))) for p in raw_x_paths]
            y_numbers = [tuple(int(nr) for nr in re.findall(r"\d+", os.path.basename(p))) for p in raw_y_paths]
            common_numbers = set(x_numbers) & set(y_numbers)
            assert len(common_numbers) > 1, (set(x_numbers), set(y_numbers))
            x_paths = numpy.array(sorted([p for p, xn in zip(raw_x_paths, x_numbers) if xn in common_numbers]))[
                n_indices
            ]
            y_paths = numpy.array(sorted([p for p, yn in zip(raw_y_paths, y_numbers) if yn in common_numbers]))[
                n_indices
            ]
            assert len(x_paths) >= 1, raw_x_paths
            assert len(x_paths) == len(y_paths), raw_y_paths
            self._len = len(x_paths)
            if save:
                tmp_data_file_name = data_file_name.as_posix().replace(".n5", "_tmp.n5")
                assert not Path(tmp_data_file_name).exists(), f"{tmp_data_file_name} exists!"
                data_file = z5py.File(path=tmp_data_file_name, mode="w", use_zarr_format=False)

                x_ds = data_file.create_dataset(
                    "x", shape=(len(x_paths),) + self.x_shape, chunks=(1,) + self.x_shape, dtype=numpy.float32
                )
                y_ds = data_file.create_dataset(
                    "y", shape=(len(self),) + self.y_shape, chunks=(1,) + self.y_shape, dtype=numpy.float32
                )

                def process_x_chunk(i):
                    ximg = imread(x_paths[i])
                    assert len(ximg.shape) == 2, ximg.shape
                    x_ds[i, ...] = resize(ximg, x_shape, roi=x_roi, order=self.interpolation_order)

                def process_y_chunk(i):
                    yimg = imread(y_paths[i])
                    assert len(yimg.shape) == 3, yimg.shape
                    y_ds[i, ...] = resize(yimg, y_shape, roi=y_roi, order=self.interpolation_order)

                # first slice in main thread to catch exceptions
                process_x_chunk(0)
                process_y_chunk(0)
                # other slices in parallel
                futures = []
                with ThreadPoolExecutor(max_workers=16) as executor:
                    for i in range(1, len(self)):
                        futures.append(executor.submit(process_x_chunk, i))
                        futures.append(executor.submit(process_y_chunk, i))

                    for fut in as_completed(futures):
                        e = fut.exception()
                        if e is not None:
                            raise e
                # for debugging: other slices in serial
                # for i in range(1, len(self)):
                #     process_x_chunk(i)
                #     process_y_chunk(i)

                os.rename(tmp_data_file_name, data_file_name.as_posix())
                self.data_file = z5py.File(path=data_file_name.as_posix())
            else:
                self.x_paths = x_paths
                self.y_paths = y_paths

        stat_path = Path(data_file_name.as_posix().replace(".n5", "_stat.yml"))
        assert ".yml" in stat_path.as_posix(), "replacing '.n5' with '_stat.yml' did not work!"
        assert len(self) >= 1, "corrupt saved dataset file?"
        logger.info("dataset length: %s", len(self))

        if interesting_paths is None:
            self.interesting_path_slices = [[]]
        else:
            self.interesting_path_slices: List[List[Optional[Tuple[slice, slice, slice, slice]]]] = [
                [] for _ in range(len(self))
            ]
            for ip in interesting_paths:
                interesting_path = numpy.array(ip.points, dtype=numpy.float)
                # scale interesting path to resized roi of target
                for i, (new_s, old_s) in enumerate(zip(y_shape[1:], original_y_shape)):
                    roi_start = y_roi[i].start
                    interesting_path[:, 1 + i] -= 0 if roi_start is None else roi_start
                    interesting_path[:, 1 + i] *= new_s
                    interesting_path[:, 1 + i] /= old_s

                path_start = int(interesting_path[:, 0].min())
                path_stop = int(interesting_path[:, 0].max())
                # interpolate missing points
                full_path = griddata(
                    points=interesting_path[:, 0],
                    values=interesting_path[:, 1:],
                    xi=numpy.arange(path_start, path_stop + 1),
                    method="linear",
                )
                # x = numpy.linspace(scipy.stats.norm.ppf(0.01), scipy.stats.norm.ppf(0.99), 100)
                # ax.plot(x, norm.pdf(x), 'r-', lw=5, alpha=0.6, label='norm pdf')
                # mask = sparse.csr_matrix(new_y_shape, dtype=numpy.float32)
                centers = numpy.round(full_path).astype(int)
                # for p_i, approx_point in enumerate(full_path):
                #     # center = [int(numpy.round(p)) for p in approx_point]
                #     roi = [slice(max(0, p-10), p+10) for p in center]
                for item in range(len(self)):
                    if item < path_start or item > path_stop:
                        self.interesting_path_slices[item].append(None)
                    else:
                        center = centers[item - path_start]
                        assert len(center.shape) == 1 and center.shape[0] == 3, center.shape
                        self.interesting_path_slices[item].append(
                            (
                                slice(None),
                                slice(max(0, center[0] - 1), center[0] + 1),
                                slice(max(0, center[1] - 3), center[1] + 3),
                                slice(max(0, center[1] - 3), center[1] + 3),
                            )
                        )

        self.transform = None
        self.has_aux = False
        self.stat = DatasetStat.load(path=stat_path, dataset=self)

        transform_instances = []
        for t in transforms:
            if isinstance(t, Transform):
                transform_instances.append(t)
            else:
                for ti in t(self.stat):
                    transform_instances.append(ti)

        self.transform = Compose(*transform_instances)
        self.has_aux = has_aux

    def __len__(self):
        return self._len

    def __getitem__(
        self, item
    ) -> Union[Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray], Tuple[numpy.ndarray, numpy.ndarray]]:
        item = int(item)
        if self.data_file is None:
            x, y = (
                resize(imread(self.x_paths[item]), self.x_shape, order=self.interpolation_order),
                resize(imread(self.y_paths[item]), self.y_shape, order=self.interpolation_order),
            )
        elif self.futures is None:
            # logger.debug("here1 x dg %s", self.data_file["x"].shape)
            x, y = self.data_file["x"][item], self.data_file["y"][item]
            # logger.debug("here2 x %s", x.shape)
            x, y = x[0], y[0]
            # logger.debug("here3 x %s", x.shape)
        else:
            raise NotImplementedError
            # todo: set high priority for "item"
            for fut in self.futures[item]:
                fut.result()

            x, y = self.data_file["x"][item], self.data_file["y"][item]

        if self.transform is not None:
            logger.debug("apply transform %s", self.transform)
            # logger.debug("x: %s, y: %s", x.shape, y.shape)
            x, y = self.transform(x, y)

        x, y = numpy.ascontiguousarray(x), numpy.ascontiguousarray(y)
        if self.has_aux:
            return x, y, numpy.array(y)
        else:
            return x, y


class DatasetFactory:
    def __init__(self, *entries: DatasetConfigEntry, n: Optional[int] = None, has_aux: bool = False):
        assert not has_aux, "deprecated"
        self.entries = entries
        self.n = n
        self.has_aux = has_aux

    def get_z_out(self) -> int:
        entry = self.entries[0]
        try:
            img_name = next(entry.y_path.glob("*.tif")).as_posix()
        except StopIteration:
            logger.error("no tif files at %s", entry.y_path.as_posix())
            raise

        y_shape = imread(img_name).shape
        z_out = y_shape[0]
        z_slice = entry.y_roi[0]
        if z_slice.stop is not None:
            assert z_out > z_slice.stop, (z_out, z_slice.stop)
            z_out = z_out - (z_out - z_slice.stop)

        if z_slice.start is not None:
            z_out -= z_slice.start

        if z_slice.step is not None and z_slice.step != 1:
            raise NotImplementedError("z slice with step !=1")

        return z_out

    def create_dataset(
        self,
        get_yx_yy: Callable[[Tuple[int, int]], Tuple[int, int]],
        transforms: List[Union[str, Transform]],
        save=True,
        n=None,
    ) -> Tuple[torch.utils.data.ConcatDataset, int, List[List[Tuple[slice, slice, slice, slice]]]]:
        if n is None:
            n = self.n

        datasets = []
        z_out = 0
        interesting_path_slices = []
        for entry in self.entries:
            try:
                img_name = next(entry.y_path.glob("*.tif")).as_posix()
            except StopIteration:
                logger.error(entry.y_path.absolute())
                raise

            y_shape = imread(img_name)[entry.y_roi].shape
            logger.info("determined shape of %s to be %s", img_name, y_shape)
            if z_out:
                assert z_out == y_shape[0], (z_out, y_shape[0])
            else:
                z_out: int = y_shape[0]

            x_img_name = next(entry.x_path.glob("*.tif")).as_posix()
            x_shape: Tuple[int, int] = imread(x_img_name)[entry.x_roi].shape
            assert y_shape[1:] == x_shape, (y_shape[1:], x_shape)

            new_y_shape = (z_out,) + get_yx_yy(x_shape)

            ds = N5Dataset(
                name=entry.description,
                x_folder=entry.x_path,
                y_folder=entry.y_path,
                x_shape=x_shape,
                y_shape=new_y_shape,
                x_roi=entry.x_roi,
                y_roi=entry.y_roi,
                transforms=transforms,
                original_y_shape=y_shape,
                interesting_paths=entry.interesting_paths,
                save=save,
                n=n,
                has_aux=self.has_aux,
            )
            interesting_path_slices += ds.interesting_path_slices
            datasets.append(ds)

        return torch.utils.data.ConcatDataset(datasets), z_out, interesting_path_slices


class Result(torch.utils.data.Dataset):
    def __init__(self, *file_paths: Path):
        self.folders = file_paths
        for folder in self.folders:
            folder.mkdir(parents=True)
        # self.file = z5py.File(path=file_path.as_posix(), mode="w", use_zarr_format=False)

    def update(self, *batches: numpy.ndarray, at: int):
        assert len(batches) == len(self.folders)
        with ThreadPoolExecutor(max_workers=8) as executor:
            for bi, batch in enumerate(batches):
                assert len(batch.shape) == 4 or len(batch.shape) == 5, batch.shape
                batch = (batch.clip(min=0, max=1) * numpy.iinfo(numpy.uint16).max).astype(numpy.uint16)
                for i, img in enumerate(batch, start=at):
                    executor.submit(imsave, (self.folders[bi] / f"{i:04.0f}.tif").as_posix(), img)


class SubsetSequentialSampler(torch.utils.data.sampler.Sampler):
    r"""Samples elements in fixed order from a given list of indices.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices: Sequence[int]):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)
