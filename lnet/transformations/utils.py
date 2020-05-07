import typing

from .base import Transform, ComposedTransformation


def get_composed_transformation_from_config(
    transformation_config: typing.List[typing.Dict[str, typing.Dict[str, typing.Any]]]
):
    assert all([len(entry) == 1 for entry in transformation_config]), "one transformation per list entry!"
    import lnet.transformations

    trf_instances: typing.List[Transform] = [
        getattr(lnet.transformations, name)(**kwargs) for trf in transformation_config for name, kwargs in trf.items()
    ]

    return ComposedTransformation(*trf_instances)
