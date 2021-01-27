import collections
from enum import Enum
from typing import Any, Dict, List, NamedTuple, Optional, OrderedDict, Sequence, Tuple, Union

import numpy
import torch.utils.data

from hylfm.datasets import ConcatDataset, TensorInfo, ZipDataset, get_dataset_from_info, get_tensor_info
from hylfm.transformations import (
    AdditiveGaussianNoise,
    Cast,
    ChannelFromLightField,
    ComposedTransformation,
    Crop,
    CropWhatShrinkDoesNot,
    Identity,
    Normalize01,
    PoissonNoise,
    RandomIntensityScale,
    RandomlyFlipAxis,
    TransformLike,
)


def identity(tensors: OrderedDict) -> OrderedDict:
    return tensors


def get_dataset_subsection(
    tensors: Dict[str, Union[str, dict]],
    indices: Optional[Union[int, slice, List[int], numpy.ndarray]] = None,
    filters: Sequence[Tuple[str, Dict[str, Any]]] = tuple(),
    preprocess_sample: Sequence[Dict[str, Dict[str, Any]]] = tuple(),
    augment_sample: TransformLike = Identity(),
) -> torch.utils.data.Dataset:
    filters = list(filters)
    infos = collections.OrderedDict()
    meta = tensors.pop("meta", {})
    for name, info_name in tensors.items():
        if isinstance(info_name, str):
            info = get_tensor_info(info_name=info_name, name=name, meta=meta)
        elif isinstance(info_name, dict):
            info = TensorInfo(name=name, **info_name)
        else:
            raise TypeError(info_name)

        trfs_for_name = [
            trf for trf in preprocess_sample if any([kwargs["apply_to"] == name for kwargs in trf.values()])
        ]
        info.transformations += trfs_for_name
        if "_repeat" in name:
            name, _ = name.split("_repeat")

        infos[name] = info

    if isinstance(indices, list):
        assert all(isinstance(i, int) for i in indices)
    elif isinstance(indices, int):
        indices = [indices]
    elif indices is None:
        indices = slice(None)
    else:
        raise NotImplementedError(indices)

    return ZipDataset(
        collections.OrderedDict(
            [
                (name, get_dataset_from_info(dsinfo, cache=True, filters=filters, indices=indices))
                for name, dsinfo in infos.items()
            ]
        ),
        transformation=augment_sample,
    )


class DatasetName(str, Enum):
    beads_sample0 = "beads_sample0"
    beads_small0 = "beads_small0"
    heart_static0 = "heart_static0"


class DatasetPart(str, Enum):
    train = "train"
    validate = "validate"
    test = "test"
    whole = "whole"


class DatasetAndTransforms(NamedTuple):
    dataset: ConcatDataset
    batch_preprocessing: TransformLike
    batch_preprocessing_in_step: TransformLike
    batch_postprocessing: TransformLike


def get_dataset(
    name: DatasetName, part: DatasetPart, nnum: int, z_out: int, scale: int, shrink: int, interpolation_order: int = 2
):
    meta = {
        "nnum": nnum,
        "z_out": z_out,
        "scale": scale,
        "interpolation_order": interpolation_order,
        "crop_names": set(),
    }
    batch_preprocessing = Identity()
    batch_preprocessing_in_step = Identity()
    batch_postprocessing = Identity()

    sections = []
    if name == DatasetName.beads_sample0:
        indices = {
            DatasetPart.whole: [0, 1, 2],
            DatasetPart.train: [0],
            DatasetPart.validate: [1],
            DatasetPart.test: [2],
        }[part]
        batch_preprocessing_in_step = Cast(apply_to=["lfc", "ls_reg"], dtype="float32", device="cuda", non_blocking=True)
        preprocess_sample = [
            {"Resize": {"apply_to": "ls_reg", "shape": [1.0, 121, scale / 19, scale / 19], "order": 2}},
            {"Assert": {"apply_to": "ls_reg", "expected_tensor_shape": [1, 121, None, None]}},
        ]
        augment_sample = (
            ComposedTransformation(
                Crop(apply_to="ls_reg", crop=((0, None), (35, -35), (shrink, -shrink), (shrink, -shrink))),
                Normalize01(apply_to="lf", min_percentile=5.0, max_percentile=99.8),
                Normalize01(apply_to="ls_reg", min_percentile=5.0, max_percentile=99.99),
                AdditiveGaussianNoise(apply_to="lf", sigma=0.1),
                AdditiveGaussianNoise(apply_to="ls_reg", sigma=0.05),
                RandomIntensityScale(apply_to=["lf", "ls_reg"], factor_min=0.8, factor_max=1.2),
                RandomlyFlipAxis(apply_to=["lf", "ls_reg"], axis=-1),
                RandomlyFlipAxis(apply_to=["lf", "ls_reg"], axis=-2),
            )
            if part == DatasetPart.train
            else ComposedTransformation(
                Crop(apply_to="ls_reg", crop=((0, None), (35, -35), (shrink, -shrink), (shrink, -shrink))),
                Normalize01(apply_to="lf", min_percentile=5.0, max_percentile=99.8),
                Normalize01(apply_to="ls_reg", min_percentile=5.0, max_percentile=99.99),
                ChannelFromLightField(apply_to={"lf": "lfc"}, nnum=nnum),
            )
        )
        sections.append(
            [
                get_dataset_subsection(
                    tensors={"lf": "local.beads.b01highc_0", "ls_reg": "local.beads.b01highc_0", "meta": meta},
                    filters=[],
                    indices=indices,
                    preprocess_sample=preprocess_sample,
                    augment_sample=augment_sample,
                )
            ]
        )
    elif name == DatasetName.beads_small0:
        batch_preprocessing_in_step = Cast(apply_to=["lfc", "ls_reg"], dtype="float32", device="cuda", non_blocking=True)
        if part in [DatasetPart.whole, DatasetPart.train]:
            if meta["scale"] != 8:
                # due to size zenodo upload is resized to scale 8
                preprocess_sample = [
                    {
                        "Resize": {
                            "apply_to": "ls_reg",
                            "shape": [1.0, 1.0, meta["scale"] / 8, meta["scale"] / 8],
                            "order": 2,
                        }
                    }
                ]
            else:
                preprocess_sample = []

            sections.append(
                [
                    get_dataset_subsection(
                        tensors={"lf": f"beads.small_2", "ls_reg": f"beads.small_2", "meta": meta},
                        filters=[],
                        indices=None,
                        preprocess_sample=preprocess_sample,
                        augment_sample=ComposedTransformation(
                            Crop(apply_to="ls_reg", crop=((0, None), (35, -35), (shrink, -shrink), (shrink, -shrink))),
                            Normalize01(apply_to="lf", min_percentile=5.0, max_percentile=99.8),
                            Normalize01(apply_to="ls_reg", min_percentile=5.0, max_percentile=99.99),
                            AdditiveGaussianNoise(apply_to="lf", sigma=0.1),
                            AdditiveGaussianNoise(apply_to="ls_reg", sigma=0.05),
                            RandomIntensityScale(apply_to=["lf", "ls_reg"], factor_min=0.8, factor_max=1.2),
                            RandomlyFlipAxis(apply_to=["lf", "ls_reg"], axis=-1),
                            RandomlyFlipAxis(apply_to=["lf", "ls_reg"], axis=-2),
                        ),
                    )
                ]
            )

        if part in [DatasetPart.whole, DatasetPart.validate, DatasetPart.test]:
            preprocess_sample = [
                {
                    "Resize": {
                        "apply_to": "ls_reg",
                        "shape": [1.0, 121, meta["scale"] / 19, meta["scale"] / 19],
                        "order": 2,
                    }
                },
                {"Assert": {"apply_to": "ls_reg", "expected_tensor_shape": [1, 121, None, None]}},
            ]
            augment_sample = ComposedTransformation(
                Crop(apply_to="ls_reg", crop=((0, None), (35, -35), (shrink, -shrink), (shrink, -shrink))),
                Normalize01(apply_to="lf", min_percentile=5.0, max_percentile=99.8),
                Normalize01(apply_to="ls_reg", min_percentile=5.0, max_percentile=99.99),
                ChannelFromLightField(apply_to={"lf": "lfc"}, nnum=nnum),
            )

            if part in [DatasetPart.whole, DatasetPart.validate]:
                sections.append(
                    [
                        get_dataset_subsection(
                            tensors={"lf": f"beads.small_0", "ls_reg": f"beads.small_0", "meta": meta},
                            filters=[],
                            indices=None,
                            preprocess_sample=preprocess_sample,
                            augment_sample=augment_sample,
                        )
                    ]
                )

            if part in [DatasetPart.whole, DatasetPart.test]:
                sections.append(
                    [
                        get_dataset_subsection(
                            tensors={"lf": f"beads.small_1", "ls_reg": f"beads.small_1", "meta": meta},
                            filters=[],
                            indices=None,
                            preprocess_sample=preprocess_sample,
                            augment_sample=augment_sample,
                        )
                    ]
                )

    elif name == DatasetName.heart_static0:
        batch_preprocessing_in_step = Cast(apply_to=["lfc", "ls_trf"], dtype="float32", device="cuda", non_blocking=True)
        preprocess_sample = []
        if part == DatasetPart.train:
            augment_sample = ComposedTransformation(
                CropWhatShrinkDoesNot(apply_to="lf", meta=meta, wrt_ref=True),
                CropWhatShrinkDoesNot(apply_to="ls_trf", meta=meta, wrt_ref=False),
                Crop(apply_to="ls_trf", crop=((0, None), (0, None), (shrink, -shrink), (shrink, -shrink))),
                Normalize01(apply_to="lf", min_percentile=5.0, max_percentile=99.8),
                Normalize01(apply_to="ls_trf", min_percentile=5.0, max_percentile=99.99),
                RandomIntensityScale(apply_to=["lf", "ls_trf"], factor_min=0.8, factor_max=1.2),
                PoissonNoise(apply_to="lf", peak=10),
                PoissonNoise(apply_to="ls_trf", peak=10),
                RandomlyFlipAxis(apply_to=["lf", "ls_trf"], axis=-1),
                RandomlyFlipAxis(apply_to=["lf", "ls_trf"], axis=-2),
            )
        else:
            augment_sample = ComposedTransformation(
                CropWhatShrinkDoesNot(apply_to="lf", meta=meta, wrt_ref=True),
                CropWhatShrinkDoesNot(apply_to="ls_trf", meta=meta, wrt_ref=False),
                Crop(apply_to="ls_trf", crop=((0, None), (0, None), (shrink, -shrink), (shrink, -shrink))),
                Normalize01(apply_to="lf", min_percentile=5.0, max_percentile=99.8),
                Normalize01(apply_to="ls_trf", min_percentile=5.0, max_percentile=99.99),
                ChannelFromLightField(apply_to={"lf": "lfc"}, nnum=nnum),
            )

        if part in [DatasetPart.whole, DatasetPart.train]:
            subsections = []
            for tag in [  # fish3
                "2019-12-10_04.24.29",
                "2019-12-10_05.14.57",
                "2019-12-10_05.41.48",
                "2019-12-10_06.03.37",
                "2019-12-10_06.25.14",
            ]:
                meta["crop_names"].add("staticHeartFOV")
                tensors = {"lf": f"heart_static.{tag}", "ls_trf": f"heart_static.{tag}", "meta": meta}
                subsections.append(
                    get_dataset_subsection(
                        tensors=tensors,
                        filters=[],
                        indices=None,
                        preprocess_sample=preprocess_sample,
                        augment_sample=augment_sample,
                    )
                )

            sections.append(subsections)

            subsections = []
            for tag in [  # fish1
                "2019-12-09_02.16.30",
                "2019-12-09_02.23.01",
                "2019-12-09_02.29.34",
                "2019-12-09_02.35.49",
                "2019-12-09_02.42.03",
                "2019-12-09_02.48.24",
                "2019-12-09_02.54.46",
            ]:
                meta["crop_names"].add("Heart_tightCrop")
                tensors = {"lf": f"heart_static.{tag}", "ls_trf": f"heart_static.{tag}", "meta": meta}
                subsections.append(
                    get_dataset_subsection(
                        tensors=tensors,
                        filters=[],
                        indices=None,
                        preprocess_sample=preprocess_sample,
                        augment_sample=augment_sample,
                    )
                )

            sections.append(subsections)

            subsections = []
            for tag in [  # fish5
                "2019-12-08_06.35.52",
                "2019-12-08_06.38.47",
                "2019-12-08_06.10.34",
                "2019-12-08_06.41.39",
                "2019-12-08_06.18.09",
                "2019-12-08_06.46.09",
                "2019-12-08_06.23.13",
                "2019-12-08_06.49.08",
                "2019-12-08_06.25.02",
                "2019-12-08_06.51.57",
                "2019-12-08_06.30.40",
            ]:
                meta["crop_names"].add("Heart_tightCrop")
                tensors = {"lf": f"heart_static.{tag}", "ls_trf": f"heart_static.{tag}", "meta": meta}
                subsections.append(
                    get_dataset_subsection(
                        tensors=tensors,
                        filters=[],
                        indices=slice(1, None, None),
                        preprocess_sample=preprocess_sample,
                        augment_sample=augment_sample,
                    )
                )

            sections.append(subsections)

    else:
        raise NotImplementedError(name)

    return DatasetAndTransforms(
        ConcatDataset([torch.utils.data.ConcatDataset(subsections) for subsections in sections]),
        batch_preprocessing,
        batch_preprocessing_in_step,
        batch_postprocessing,
    )
