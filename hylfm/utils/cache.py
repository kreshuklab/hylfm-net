import json
from functools import wraps

import pickle
from hashlib import sha256
from pathlib import Path
from typing import Tuple, Union

import numpy
from numpy.lib.npyio import NpzFile

from lnet import settings


def load_with_pickle_or_numpy(path: Path):
    if path.suffix == ".pickle":
        with path.open("rb") as f:
            return pickle.load(f)
    elif path.suffix == ".npz":
        npzf: NpzFile = numpy.load(str(path))
        npz_keys = list(npzf.keys())
        if len(npz_keys) != 1:
            raise NotImplementedError(npz_keys)

        ret = npzf[npz_keys[0]]
        npzf.close()
        return ret
    else:
        raise NotImplementedError(path.suffix)


def save_with_pickle_or_numpy(obj, path: Path, _nested=False) -> Union[Path, Tuple[Path, ...]]:
    assert not path.suffix
    if isinstance(obj, numpy.ndarray):
        path = path.with_suffix(".npz")
        numpy.savez_compressed(str(path.with_suffix(".npz")), path)
    elif isinstance(obj, tuple) and not _nested:
        path = tuple(
            [save_with_pickle_or_numpy(b, path.with_name(path.name + f"_{i}"), _nested=True) for i, b in enumerate(obj)]
        )
    else:
        path = path.with_suffix(".pickle")
        with path.open("wb") as f:
            pickle.dump(obj, f)

    return path


def cached_to_disk(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        cache_key = sha256()

        def update_cache_key(obj):
            if isinstance(obj, numpy.ndarray):
                cache_key.update(obj.data.to_bytes())
            else:
                if isinstance(obj, Path):
                    obj = str(obj)

                cache_key.update(json.dumps(obj, sort_keys=True).encode("utf-8"))

        for arg in args:
            update_cache_key(arg)

        for k, v in sorted(kwargs.items()):
            update_cache_key(k)
            update_cache_key(v)

        cache_key = cache_key.hexdigest()

        result_description_path = (
                settings.cache_dir / cached_to_disk.__name__ / func.__name__ / f"{cache_key}.descr.pickle"
        )

        try:
            with result_description_path.open("rb") as f:
                result_description = pickle.load(f)

            if isinstance(result_description, Path):
                return load_with_pickle_or_numpy(result_description)
            elif isinstance(result_description, tuple):
                return tuple([load_with_pickle_or_numpy(p) for p in result_description])
            else:
                raise NotImplementedError(type(result_description))
        except FileNotFoundError:
            result = func(*args, **kwargs)
            result_description_path.parent.mkdir(parents=True, exist_ok=True)
            result_description = save_with_pickle_or_numpy(
                result, result_description_path.with_suffix("").with_suffix("")
            )
            with result_description_path.open("wb") as f:
                pickle.dump(result_description, f)

            return result

    return wrapper
