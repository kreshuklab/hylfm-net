import re
import shutil
import typing
from functools import wraps
from inspect import signature
from pathlib import Path
from time import perf_counter

import requests
import torch
from merge_args import merge_args
from tqdm import tqdm


def return_unused_kwargs_to(fn):
    @merge_args(fn)
    def fn_return_unused_kwargs(**kwargs):
        used_kwargs = {key: kwargs.pop(key) for key in signature(fn).parameters if key in kwargs}
        return fn(**used_kwargs), kwargs

    return fn_return_unused_kwargs


def camel_to_snake(name: str) -> str:
    name = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", name).lower()


def rename(newname):
    def decorator(f):
        f.__name__ = newname
        return f

    return decorator


def handle_tensors(*output_names: str, **kwarg_mapping):
    def decorator(function):
        @wraps(function)
        def wrapper(tensors: typing.OrderedDict[str, typing.Any]) -> typing.OrderedDict[str, typing.Any]:
            kwargs = {name: tensors[tensor_name] for name, tensor_name in kwarg_mapping.items()}
            outputs = function(**kwargs)
            if not output_names:
                pass
            elif len(output_names) == 1:
                tensors[output_names[0]] = outputs
            else:
                assert len(output_names) == len(outputs)
                for name, out in zip(output_names, outputs):
                    tensors[name] = out

            return tensors

        return wrapper

    return decorator


def delete_empty_dirs(dir: Path):
    if dir.is_dir():
        for d in dir.iterdir():
            delete_empty_dirs(d)

        if not any(dir.rglob("*")):
            dir.rmdir()


def percentile(t: torch.Tensor, q: float) -> typing.Union[int, float]:
    """
    from: https://gist.github.com/spezold/42a451682422beb42bc43ad0c0967a30#file-torch_percentile-py-L7
    Return the ``q``-th percentile of the flattened input tensor's data.

    CAUTION:
     * Needs PyTorch >= 1.1.0, as ``torch.kthvalue()`` is used.
     * Values are not interpolated, which corresponds to
       ``numpy.percentile(..., interpolation="nearest")``.

    :param t: Input tensor.
    :param q: Percentile to compute, which must be between 0 and 100 inclusive.
    :return: Resulting value (scalar).
    """
    # Note that ``kthvalue()`` works one-based, i.e. the first sorted value
    # indeed corresponds to k=1, not k=0! Use float(q) instead of q directly,
    # so that ``round()`` returns an integer, even if q is a np.float32.
    k = 1 + round(0.01 * float(q) * (t.numel() - 1))
    result = t.view(-1).kthvalue(k).values.item()
    return result


def print_timing(func):
    """
    create a timing decorator function
    use
    @print_timing
    just above the function you want to time
    """

    @wraps(func)  # improves debugging
    def wrapper(*args, **kwargs):
        start = perf_counter()  # needs python3.3 or higher
        result = func(*args, **kwargs)
        print(f"{func.__name__} took {(perf_counter() - start) * 1000:.3f} ms")
        return result

    return wrapper


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
