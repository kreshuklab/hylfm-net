import re
import typing
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from time import perf_counter

import torch


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
