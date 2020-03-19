from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from .model import ModelConfig
from lnet.transforms import known_transforms


def get_known(known: Dict[str, Any], name: str):
    res = known.get(name, None)
    if res is None:
        raise ValueError(f"{name} not known. Valid values are:\n{', '.join(known.keys())}")

    return res


def resolve_python_name_conflicts(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    trailing_underscore = ["min", "max"]
    return {k + "_" if k in trailing_underscore else k: v for k, v in kwargs.items()}


def get_trfs_and_their_names(
    model_config: ModelConfig, transforms: Optional[List[Any]], conf: Optional[List[Union[str, Dict[str, Any]]]]
) -> Tuple[List[Callable], List[str]]:
    if transforms is None:
        if conf is None:
            new_transforms = []
            new_names = []
        else:
            new_transforms = []
            new_names = []
            for c in conf:
                if isinstance(c, str):
                    name = c
                    kwargs = {}
                else:
                    assert isinstance(c, dict), type(c)
                    name = c["name"]
                    kwargs = c.get("kwargs", {})
                    left = {k: v for k, v in c.items() if k not in ["name", "kwargs"]}
                    if left:
                        raise ValueError(
                            f"invalid keys in transformation entry with name: {name} and kwargs: {kwargs}: " f"{left}"
                        )

                kwargs = resolve_python_name_conflicts(kwargs)
                new_transforms.append(get_known(known_transforms, name)(model_config=model_config, kwargs=kwargs))
                new_names.append(name)

    elif conf is None:
        new_transforms = transforms
        new_names = [t.__name__ for t in transforms]
    else:
        new_transforms = transforms
        new_names = [c if isinstance(c, str) else c["name"] for c in conf]

    return new_transforms, new_names
