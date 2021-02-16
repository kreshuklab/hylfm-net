import logging
import shutil
from pathlib import Path
from typing import Union

import numpy
import pandas
import requests
import torch
from tifffile import imwrite
from tqdm import tqdm

logger = logging.getLogger(__name__)


def save_tensor(path: Path, tensor: Union[numpy.ndarray, torch.Tensor]):
    assert path.suffix in (".tif", ".tiff"), path.suffix

    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()

    tensor = numpy.moveaxis(tensor, 0, -1)  # save channel last
    *zyx, c = tensor.shape
    assert len(zyx) in (2, 3), zyx
    assert c in (1, 3, 361), c
    tensor = tensor.squeeze()
    if c == 1:
        tensor = tensor[..., None]

    dtype_map = {"float64": "float32"}
    tensor = tensor.astype(dtype_map.get(str(tensor.dtype), tensor.dtype), copy=False)

    tif_kwargs = {"compress": 2}
    try:
        imwrite(str(path), tensor, **tif_kwargs)
    except Exception as e:
        logger.error(e, exc_info=True)
        imwrite(str(path), tensor, **tif_kwargs, bigtiff=True)


def download_file_from_zenodo(doi: str, file_name: str, download_file_path: Path):
    url = "https://doi.org/" + doi
    r = requests.get(url)
    if not r.ok:
        raise RuntimeError("DOI could not be resolved.")

    zenodo_record_id = r.url.split("/")[-1]

    return download_file(
        url=f"https://zenodo.org/record/{zenodo_record_id}/files/{file_name}", download_file_path=download_file_path
    )


def download_file(url: str, download_file_path: Path):
    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get("content-length", 0))
    block_size = 1024  # 1 Kibibyte
    progress_bar = tqdm(
        total=total_size_in_bytes, unit="iB", unit_scale=True, desc=f"Downloading {download_file_path.name}"
    )
    download_file_path.parent.mkdir(parents=True, exist_ok=True)
    with download_file_path.with_suffix(".part").open("wb") as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)

    progress_bar.close()
    # todo: checksum
    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        raise RuntimeError(f"downloading {url} to {download_file_path} failed")

    shutil.move(download_file_path.with_suffix(".part"), download_file_path)


def save_pandas_df(df: pandas.DataFrame, df_path: Path):
    if not df_path.suffix:
        df_path = df_path.with_suffix(".h5")

    if ".h5" in df_path.suffix or ".hdf5" in df_path.suffix:
        df_path, *internal_h5_path = df_path.name.split("/")
        if not internal_h5_path:
            internal_h5_path = ["df"]

        store = pandas.HDFStore(str(df_path))
        store["/".join(internal_h5_path)] = df
    elif df_path.suffix in (".pkl", ".pickle"):
        df.to_pickle(str(df_path))
    else:
        raise NotImplementedError(df_path)
