import inspect
import logging
from contextlib import suppress
from dataclasses import InitVar
from functools import wraps
from typing import List, Optional, Union, _SpecialForm, ClassVar, Any

logger = logging.getLogger(__name__)


def enforce_types(callable):
    spec = inspect.getfullargspec(callable)

    def check_type(name, value, expected_type):
        if expected_type is None or str(expected_type) == "NoneType":
            ok = expected_type is None
        elif isinstance(expected_type, _SpecialForm):
            raise NotImplementedError(expected_type)
            # if hasattr(expected_type, "__args__"):
            # else:
            #     assert expected_type is not Union
            #     assert expected_type is Any or expected_type is Union or expected_type is ClassVar
            #     raise NotImplementedError(expected_type)
        elif hasattr(expected_type, "__origin__"):
            # In Python 3.8: actual_type = typing.get_origin(expected_type) or expected_type
            if expected_type.__origin__ is Union or expected_type.__origin__ is ClassVar:
                return check_type(name, value, expected_type.__args__)
            else:
                return check_type(name, value, expected_type.__origin__)
        elif hasattr(expected_type, "__args__"):
            raise NotImplementedError(expected_type)
        elif expected_type is InitVar:
            return
        elif isinstance(expected_type, tuple):
            ok = False
            for candidate in expected_type:
                try:
                    check_type(name, value, candidate)
                except:
                    pass
                else:
                    ok = True
        else:
            try:
                ok = isinstance(value, expected_type)
            except Exception as e:
                logger.error(e)
                raise NotImplementedError(expected_type)

        if not ok:
            raise TypeError(
                "Unexpected type for '{}' (expected {} but found {})".format(name, expected_type, type(value))
            )

    def check_types(*args, **kwargs):
        parameters = dict(zip(spec.args, args))
        parameters.update(kwargs)
        for name, value in parameters.items():
            try:
                type_hint = spec.annotations[name]
            except KeyError:
                pass
            else:
                check_type(name, value, type_hint)

    def decorate(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            check_types(*args, **kwargs)
            return func(*args, **kwargs)

        return wrapper

    if inspect.isclass(callable):
        callable.__init__ = decorate(callable.__init__)
        return callable

    return decorate(callable)


def range_or_single_index_to_list(indice_string_part: str) -> List[Optional[int]]:
    """
    :param indice_string_part: e.g. 37 or 0-100 or 0-100-10
    :return: e.g. [37] or [0, 1, 2, ..., 99] or [0, 10, 20, ..., 90]
    """

    ints_in_part = [int(p) for p in indice_string_part.split("-") if p]
    assert len(ints_in_part) < 4, ints_in_part
    return list(range(*ints_in_part)) if "-" in indice_string_part else ints_in_part


def indice_string_to_list(indice_string: Optional[Union[str, int]]) -> Optional[List[int]]:
    if indice_string is None:
        return None
    elif isinstance(indice_string, int):
        return [indice_string]
    else:
        concatenated_indices: List[int] = []
        for part in indice_string.split("|"):
            concatenated_indices += range_or_single_index_to_list(part)

        return concatenated_indices
