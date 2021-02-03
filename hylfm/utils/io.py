import logging
from pathlib import Path
from typing import Union

import numpy
import torch
from tifffile import imwrite

logger = logging.getLogger(__name__)


def save_tensor(path: Path, tensor: Union[numpy.ndarray, torch.Tensor]):
    assert path.suffix in (".tif", ".tiff"), path.suffix

    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()

    tensor = numpy.moveaxis(tensor, 0, -1)  # save channel last
    *zyx, c = tensor.shape
    assert len(zyx) in (2, 3), zyx
    assert c in (1, 3), c
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
