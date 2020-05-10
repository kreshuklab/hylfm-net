import typing

from .base import Transform, ComposedTransformation


def get_composed_transformation_from_config(
    transformation_config: typing.List[typing.Dict[str, typing.Dict[str, typing.Any]]]
):
    assert all([len(entry) == 1 for entry in transformation_config]), "one transformation per list entry!"
    import lnet.transformations

    trf_instances: typing.List[Transform] = []
    for trf in transformation_config:
        for name, kwargs in trf.items():
            try:
                instance = getattr(lnet.transformations, name)(**kwargs)
            except TypeError as e:
                raise type(e)(str(e) + f" for transformation {name}") from e
            else:
                trf_instances.append(instance)

    return ComposedTransformation(*trf_instances)
