import typing
from functools import wraps


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
