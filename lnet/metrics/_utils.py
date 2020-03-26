import typing


def get_output_transform(tensor_names: typing.Dict[str, str]):
    if "y_pred" in tensor_names and "y" in tensor_names and len(tensor_names) == 2:

        def output_transform(out: typing.OrderedDict[str, typing.Any]):
            return out[tensor_names["y_pred"]], out[tensor_names["y"]]

    else:

        def output_transform(out: typing.OrderedDict[str, typing.Any]):
            return {name: out[tensor_name] for name, tensor_name in tensor_names.items()}

    return output_transform
