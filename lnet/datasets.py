import logging
import warnings

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

from lnet import models
from lnet.config.model import ModelConfig
from scipy.ndimage import zoom
from tifffile import imread, imsave
from typing import List, Optional, Tuple, Union, Callable, Sequence, Generator

from lnet.config.dataset import NamedDatasetInfo
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
    x_paths: List[str]
    y_paths: Optional[List[str]]
    z_out: Optional[int]

    data_file: Optional[z5py.File] = None
    futures: Optional[List[Tuple[Future, Future]]] = None

    def __init__(
        self,
        info: NamedDatasetInfo,
        scaling: Optional[Tuple[float, float]],
        z_out: int,
        interpolation_order: int = 1,
        save: bool = True,
        transforms: Optional[List[Union[Transform, Callable[[DatasetStat], Generator[Transform, None, None]]]]] = None,
        data_folder: Optional[Path] = None,
        model_config: Optional[ModelConfig] = None,
    ):
        assert scaling is not None or model_config is not None
        super().__init__()
        name = info.description
        x_folder = info.x_path
        y_folder = info.y_path
        self.info = info

        def split_off_glob(path: Path) -> Tuple[Path, str]:
            not_glob = path.as_posix().split("*")[0]
            valid_path = Path(not_glob[: not_glob.rfind("/")])
            glob_str = path.relative_to(valid_path).as_posix()
            return valid_path, glob_str

        if x_folder.exists():
            x_glob = "*.tif"
        elif "*" in x_folder.as_posix():
            x_folder, x_glob = split_off_glob(x_folder)
            logger.info("split x_folder into %s and %s", x_folder, x_glob)
        else:
            raise NotImplementedError

        assert x_folder.exists(), x_folder.absolute()
        x_img_name = next(x_folder.glob(x_glob)).as_posix()
        if info.x_shape is None:
            original_x_shape: Tuple[int, int] = imread(x_img_name)[info.x_roi].shape
            logger.info("determined x shape of %s to be %s", x_img_name, original_x_shape)
        else:
            original_x_shape = info.x_shape

        self.z_out = z_out
        x_shape = (1,) + original_x_shape
        if y_folder:
            self.with_target = True
            if y_folder.exists():
                y_glob = "*.tif"
            elif "*" in y_folder.as_posix():
                y_folder, y_glob = split_off_glob(y_folder)
                logger.info("split y_folder into %s and %s", y_folder, y_glob)
            else:
                raise NotImplementedError

            assert y_folder.exists(), y_folder.absolute()
            try:
                img_name = next(y_folder.glob(y_glob)).as_posix()
            except StopIteration:
                logger.error(y_folder.absolute())
                raise

            if info.y_shape is None:
                original_y_shape = imread(img_name)[info.y_roi].shape
                logger.info("determined y shape of %s to be %s", img_name, original_y_shape)
                assert original_y_shape[1:] == original_x_shape, (original_y_shape[1:], original_x_shape)
            else:
                original_y_shape = info.y_shape

            if scaling is None:
                assert model_config is not None
                model_scaling = getattr(models, model_config.name)(
                    nnum=model_config.nnum, z_out=z_out, **model_config.kwargs
                ).get_scaling(original_x_shape)
                scaling = (model_scaling[0] / model_config.nnum, model_scaling[1] / model_config.nnum)

            y_shape = (self.z_out,) + tuple([int(oxs * sc) for oxs, sc in zip(original_x_shape, scaling)])
            if info.AffineTransform is not None:
                assert info.y_shape is not None, "when working with transforms the output shape needs to be given"
                scaling_matrix = numpy.zeros((4, 4), dtype=numpy.float32)
                assert len(y_shape) == len(original_y_shape) == 3
                scaling_matrix[3, 3] = 1
                for i, (out, orig) in enumerate(zip(y_shape, original_y_shape)):
                    scaling_matrix[i, i] = out / orig

                self.affine_transform = info.AffineTransform(
                    order=interpolation_order, output_shape=y_shape, additional_transforms_left=[scaling_matrix]
                )
            else:
                self.affine_transform = None

            y_shape = (1,) + y_shape  # add channel dim
        else:
            self.with_target = False
            y_shape = None

        if data_folder is None:
            data_folder = os.environ.get("DATA_FOLDER", None)
            if data_folder is None:
                data_folder = Path(__file__).parent.parent / "data"
            else:
                data_folder = Path(data_folder)

        assert data_folder.exists(), data_folder.absolute()

        self.x_shape = x_shape
        self.y_shape = y_shape
        self.interpolation_order = interpolation_order
        shapestr = (
            f"interpolation order: {interpolation_order}\n"
            f"x roi: {info.x_roi}\ny roi: {info.y_roi}\n"
            f"x shape: {x_shape}\ny shape: {y_shape}\n"
            f"x_folder: {x_folder}\ny_folder: {y_folder}"
        )
        if info.AffineTransform is not None:
            shapestr += f"\naffine transform: {info.AffineTransform.__name__}"

        data_file_name = data_folder / f"{name}_{hash(shapestr.encode()).hexdigest()}.n5"
        with Path(data_file_name.as_posix().replace(".n5", ".txt")).open("w") as f:
            f.write(shapestr)

        def get_paths_and_numbers(folder: Path, glob_expr: str):
            found_paths = list(Path(folder).glob(glob_expr))
            glob_numbers = [nr for nr in re.findall(r"\d+", glob_expr)]
            logger.info("found numbers %s in glob_exp %s", glob_numbers, glob_expr)
            numbers = [
                tuple(int(nr) for nr in re.findall(r"\d+", p.relative_to(folder).as_posix()) if nr not in glob_numbers)
                for p in found_paths
            ]
            logger.info("found number tuples %s in folder %s", numbers, folder)
            return [p.as_posix() for p in found_paths], numbers

        logger.info("data_file_name %s", data_file_name)
        if data_file_name.exists():
            self.data_file = z5py.File(path=data_file_name.as_posix(), mode="r", use_zarr_format=False)
            self._len = self.data_file["x"].shape[0]
        else:
            raw_x_paths, x_numbers = get_paths_and_numbers(x_folder, x_glob)
            common_numbers = set(x_numbers)
            if self.with_target:
                raw_y_paths, y_numbers = get_paths_and_numbers(y_folder, y_glob)
                common_numbers &= set(y_numbers)
                assert len(common_numbers) > 1 or len(set(x_numbers) | set(y_numbers)) == 1, (
                    "x",
                    set(x_numbers),
                    "y",
                    set(y_numbers),
                    "x|y",
                    set(x_numbers) | set(y_numbers),
                    "x&y",
                    set(x_numbers) & set(y_numbers),
                )
                self.y_paths = sorted([p for p, yn in zip(raw_y_paths, y_numbers) if yn in common_numbers])

            self.x_paths = sorted([p for p, xn in zip(raw_x_paths, x_numbers) if xn in common_numbers])
            if self.with_target:
                assert len(self.x_paths) == len(self.y_paths), raw_y_paths

            assert len(self.x_paths) >= 1, raw_x_paths
            self._len = len(self.x_paths)
            if save:
                tmp_data_file_name = data_file_name.as_posix().replace(".n5", "_tmp.n5")
                assert not Path(tmp_data_file_name).exists(), f"{tmp_data_file_name} exists!"
                data_file = z5py.File(path=tmp_data_file_name, mode="w", use_zarr_format=False)

                self.x_ds = data_file.create_dataset(
                    "x", shape=(len(self.x_paths),) + self.x_shape, chunks=(1,) + self.x_shape, dtype=numpy.float32
                )
                if self.with_target:
                    self.y_ds = data_file.create_dataset(
                        "y", shape=(len(self),) + self.y_shape, chunks=(1,) + self.y_shape, dtype=numpy.float32
                    )

                # first slice in main thread to catch exceptions
                self.process_x(0, to_ds=True)
                if self.with_target:
                    self.process_y(0, to_ds=True)

                # other slices in parallel
                futures = []
                with ThreadPoolExecutor(max_workers=16) as executor:
                    for i in range(1, len(self)):
                        futures.append(executor.submit(self.process_x, i, to_ds=True))
                        if self.with_target:
                            futures.append(executor.submit(self.process_y, i, to_ds=True))

                    for fut in as_completed(futures):
                        e = fut.exception()
                        if e is not None:
                            raise e
                # for debugging: other slices in serial
                # for i in range(1, len(self)):
                #     process_x(i)
                #     process_y(i)

                os.rename(tmp_data_file_name, data_file_name.as_posix())
                self.data_file = z5py.File(path=data_file_name.as_posix())

        stat_path = Path(data_file_name.as_posix().replace(".n5", "_stat.yml"))
        assert ".yml" in stat_path.as_posix(), "replacing '.n5' with '_stat.yml' did not work!"
        assert len(self) >= 1, "corrupt saved dataset file?"
        logger.info("dataset length: %s", len(self))

        # if interesting_paths is None:
        #     self.interesting_path_slices = [[]]
        # else:
        #     self.interesting_path_slices: List[List[Optional[Tuple[slice, slice, slice, slice]]]] = [
        #         [] for _ in range(len(self))
        #     ]
        #     for ip in interesting_paths:
        #         interesting_path = numpy.array(ip.points, dtype=numpy.float)
        #         # scale interesting path to resized roi of target
        #         for i, (new_s, old_s) in enumerate(zip(y_shape[1:], original_y_shape)):
        #             roi_start = y_roi[i].start
        #             interesting_path[:, 1 + i] -= 0 if roi_start is None else roi_start
        #             interesting_path[:, 1 + i] *= new_s
        #             interesting_path[:, 1 + i] /= old_s
        #
        #         path_start = int(interesting_path[:, 0].min())
        #         path_stop = int(interesting_path[:, 0].max())
        #         # interpolate missing points
        #         full_path = griddata(
        #             points=interesting_path[:, 0],
        #             values=interesting_path[:, 1:],
        #             xi=numpy.arange(path_start, path_stop + 1),
        #             method="linear",
        #         )
        #         # x = numpy.linspace(scipy.stats.norm.ppf(0.01), scipy.stats.norm.ppf(0.99), 100)
        #         # ax.plot(x, norm.pdf(x), 'r-', lw=5, alpha=0.6, label='norm pdf')
        #         # mask = sparse.csr_matrix(new_y_shape, dtype=numpy.float32)
        #         centers = numpy.round(full_path).astype(int)
        #         # for p_i, approx_point in enumerate(full_path):
        #         #     # center = [int(numpy.round(p)) for p in approx_point]
        #         #     roi = [slice(max(0, p-10), p+10) for p in center]
        #         for item in range(len(self)):
        #             if item < path_start or item > path_stop:
        #                 self.interesting_path_slices[item].append(None)
        #             else:
        #                 center = centers[item - path_start]
        #                 assert len(center.shape) == 1 and center.shape[0] == 3, center.shape
        #                 self.interesting_path_slices[item].append(
        #                     (
        #                         slice(None),
        #                         slice(max(0, center[0] - 1), center[0] + 1),
        #                         slice(max(0, center[1] - 3), center[1] + 3),
        #                         slice(max(0, center[1] - 3), center[1] + 3),
        #                     )
        #                 )

        self.transform = None
        self.stat = DatasetStat(path=stat_path, dataset=self)
        transform_instances = []
        for t in transforms:
            if isinstance(t, Transform):
                transform_instances.append(t)
            else:
                for ti in t(self.stat):
                    transform_instances.append(ti)

        self.transform = Compose(*transform_instances)

    def process_x(self, i, to_ds=False) -> Optional[numpy.ndarray]:
        x_img = imread(self.x_paths[i])
        assert len(x_img.shape) == 2, x_img.shape
        x_img = x_img[self.info.x_roi]
        assert x_img.shape == self.x_shape[1:]
        if to_ds:
            self.x_ds[i, ...] = x_img
        else:
            return x_img

    def process_y(self, i, to_ds=False) -> Optional[numpy.ndarray]:
        y_img = imread(self.y_paths[i])
        assert len(y_img.shape) == 3, y_img.shape
        if self.affine_transform is None:
            y_img = resize(y_img, self.y_shape, roi=self.info.y_roi, order=self.interpolation_order)
        else:
            y_img = self.affine_transform.apply(y_img)
        if to_ds:
            self.y_ds[i, ...] = y_img
        else:
            return y_img

    def __len__(self):
        return self._len

    def __getitem__(
        self, item
    ) -> Union[Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray], Tuple[numpy.ndarray, numpy.ndarray], numpy.ndarray]:
        item = int(item)
        if self.data_file is None:
            x = self.process_x(item)
            if self.with_target:
                y = self.process_y(item)
                y = resize(imread(self.y_paths[item]), self.y_shape, order=self.interpolation_order)
        elif self.futures is None:
            logger.debug("here1 x dg %s", self.data_file["x"].shape)
            x = self.data_file["x"][item]
            logger.debug("here x %s", x.shape)
            x = x[0]
            if self.with_target:
                y = self.data_file["y"][item]
                y = y[0]
            # logger.debug("here2 x %s", x.shape)
        else:
            raise NotImplementedError
            # todo: set high priority for "item"
            for fut in self.futures[item]:
                fut.result()

            x = self.data_file["x"][item]
            if self.with_target:
                y = self.data_file["y"][item]

        if self.transform is not None:
            logger.debug("apply transform %s", self.transform)
            # logger.debug("x: %s, y: %s", x.shape, y.shape)
            if self.with_target:
                x, y = self.transform(x, y)
            else:
                x = self.transform(x)

        x = numpy.ascontiguousarray(x)
        if self.with_target:
            y = numpy.ascontiguousarray(y)
            return x, y
        else:
            return x


class Result(torch.utils.data.Dataset):
    folder: List[List[Path]]

    def __init__(self, *file_paths: Path, subfolders: Sequence[str] = ("",), cumsum: Sequence[int] = (float("inf"),)):
        assert len(subfolders) == len(cumsum)
        if len(subfolders) > 1:
            self.folders = [[fp / sf for fp in file_paths] for sf in subfolders]
        else:
            self.folders = [file_paths]

        self.cumsum = cumsum
        for folder_group in self.folders:
            for folder in folder_group:
                folder.mkdir(parents=True, exist_ok=False)

        # self.file = z5py.File(path=file_path.as_posix(), mode="w", use_zarr_format=False)

    def update(self, *batches: numpy.ndarray, at: int):
        batches = [b for b in batches if b.shape != (1,)]
        assert len(batches) == len(self.folders[0]), (len(batches), len(self.folders[0]))
        with ThreadPoolExecutor(max_workers=8) as executor:
            for bi, batch in enumerate(batches):
                assert len(batch.shape) == 4 or len(batch.shape) == 5, batch.shape
                # batch = (batch.clip(min=0, max=1) * numpy.iinfo(numpy.uint16).max).astype(numpy.uint16)
                for i, img in enumerate(batch, start=at):
                    ds_idx = numpy.searchsorted(self.cumsum, i, side="right").item()
                    offset = self.cumsum[ds_idx - 1] if ds_idx else 0
                    executor.submit(imsave, (self.folders[ds_idx][bi] / f"{i - offset:04.0f}.tif").as_posix(), img)
