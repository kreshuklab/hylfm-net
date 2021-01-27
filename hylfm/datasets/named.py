import collections
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union, OrderedDict

import torch.utils.data

from hylfm.datasets import ConcatDataset, TensorInfo, ZipDataset, get_dataset_from_info, get_tensor_info
from hylfm.datasets.utils import indice_string_to_list
from hylfm.transformations.utils import get_composed_transformation_from_config


def identity(tensors: OrderedDict) -> OrderedDict:
    return tensors


def get_dataset_section(
    tensors: Dict[str, Union[str, dict]],
    indices: Optional[Union[str, int, List[int]]] = None,
    filters: Sequence[Tuple[str, Dict[str, Any]]] = tuple(),
    preprocess_sample: Sequence[Dict[str, Dict[str, Any]]] = tuple(),
    augment_sample: Sequence[Dict[str, Dict[str, Any]]] = tuple(),
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
        indices = indices
    elif isinstance(indices, int):
        indices = [indices]
    elif isinstance(indices, str):
        indices = indice_string_to_list(indices)
    elif indices is None:
        pass
    else:
        raise NotImplementedError(indices)

    return ZipDataset(
        collections.OrderedDict(
            [
                (name, get_dataset_from_info(dsinfo, cache=True, filters=filters, indices=indices))
                for name, dsinfo in infos.items()
            ]
        ),
        transformation=get_composed_transformation_from_config(list(augment_sample)),
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


def get_dataset(
    name: DatasetName,
    part: DatasetPart,
    meta: Dict[str, Any],
    augment_sample: Sequence[Dict[str, Dict[str, Any]]] = tuple(),
):
    assert "nnum" in meta
    assert "z_out" in meta
    assert "scale" in meta

    if "interpolation_order" not in meta:
        meta["interpolation_order"] = 2

    assert "crop_names" not in meta

    meta["crop_names"] = set()  # [Heart_tightCrop, wholeFOV, staticHeartFOV]

    sections = []
    if name == DatasetName.beads_sample0:
        filters = []
        indices = {
            DatasetPart.whole: [0, 1, 2],
            DatasetPart.train: [0],
            DatasetPart.validate: [1],
            DatasetPart.test: [2],
        }[part]
        preprocess_sample = [
            {"Resize": {"apply_to": "ls_reg", "shape": [1.0, 121, meta["scale"] / 19, meta["scale"] / 19], "order": 2}},
            {"Assert": {"apply_to": "ls_reg", "expected_tensor_shape": [1, 121, None, None]}},
        ]
        sections.append(
            get_dataset_section(
                tensors={"lf": "local.beads.b01highc_0", "ls_reg": "local.beads.b01highc_0", "meta": meta},
                filters=filters,
                indices=indices,
                preprocess_sample=preprocess_sample,
                augment_sample=augment_sample,
            )
        )
    elif name == DatasetName.beads_small0:
        filters = []
        indices = None
        if part in [DatasetPart.whole, DatasetPart.train]:
            if meta["scale"] != 8:
                # due to size zenodo upload is resized to scale 8
                preprocess_sample = [{"Resize": {"apply_to": "ls_reg", "shape": [1.0, 1.0, meta["scale"] / 8, meta["scale"] / 8], "order": 2}}]
            else:
                preprocess_sample = []

            sections.append(
                get_dataset_section(
                    tensors={"lf": f"beads.small_2", "ls_reg": f"beads.small_2", "meta": meta},
                    filters=filters,
                    indices=indices,
                    preprocess_sample=preprocess_sample,
                    augment_sample=augment_sample,
                )
            )

        if part in [DatasetPart.whole, DatasetPart.validate, DatasetPart.test]:
            preprocess_sample = [
                {"Resize": {"apply_to": "ls_reg", "shape": [1.0, 121, meta["scale"] / 19, meta["scale"] / 19], "order": 2}},
                {"Assert": {"apply_to": "ls_reg", "expected_tensor_shape": [1, 121, None, None]}},
            ]
            if part in [DatasetPart.whole, DatasetPart.validate]:
                sections.append(
                    get_dataset_section(
                        tensors={"lf": f"beads.small_0", "ls_reg": f"beads.small_0", "meta": meta},
                        filters=filters,
                        indices=indices,
                        preprocess_sample=preprocess_sample,
                        augment_sample=augment_sample,
                    )
                )

            if part in [DatasetPart.whole, DatasetPart.test]:
                sections.append(
                    get_dataset_section(
                        tensors={"lf": f"beads.small_1", "ls_reg": f"beads.small_1", "meta": meta},
                        filters=filters,
                        indices=indices,
                        preprocess_sample=preprocess_sample,
                        augment_sample=augment_sample,
                    )
                )

    elif name == DatasetName.heart_static0:
        filters = []
        indices = None
        preprocess_sample = []

        if part in [DatasetPart.whole, DatasetPart.train]:
            for tag in [  # fish3
                "2019-12-10_04.24.29",
                # "2019-12-10_05.14.57",
                # "2019-12-10_05.41.48",
                # "2019-12-10_06.03.37",
                # "2019-12-10_06.25.14",
            ]:
                meta["crop_names"].add("staticHeartFOV")
                tensors = {"lf": f"heart_static.{tag}", "ls_trf": f"heart_static.{tag}", "meta": meta}
                sections.append(
                    get_dataset_section(
                        tensors=tensors,
                        filters=filters,
                        indices=indices,
                        preprocess_sample=preprocess_sample,
                        augment_sample=augment_sample,
                    )
                )
        """
            - 
              augment_sample: *train_sample_prep
              filters: []
              datasets:
                - {tensors: {lf: heart_static.2019-12-10_04.24.29, ls_trf: heart_static.2019-12-10_04.24.29, meta: *meta}} # fish3
                - {tensors: {lf: heart_static.2019-12-10_05.14.57, ls_trf: heart_static.2019-12-10_05.14.57, meta: *meta}} # fish3
                - {tensors: {lf: heart_static.2019-12-10_05.41.48, ls_trf: heart_static.2019-12-10_05.41.48, meta: *meta}} # fish3
                - {tensors: {lf: heart_static.2019-12-10_06.03.37, ls_trf: heart_static.2019-12-10_06.03.37, meta: *meta}} # fish3
                - {tensors: {lf: heart_static.2019-12-10_06.25.14, ls_trf: heart_static.2019-12-10_06.25.14, meta: *meta}} # fish3
            - transform_sample: *sample_trfs
              augment_sample: *train_sample_prep
              filters: []
              datasets:
                - {tensors: {lf: heart_static.2019-12-09_02.16.30, ls_trf: heart_static.2019-12-09_02.16.30, meta: *meta}} # fish1
                - {tensors: {lf: heart_static.2019-12-09_02.23.01, ls_trf: heart_static.2019-12-09_02.23.01, meta: *meta}} # fish1
                - {tensors: {lf: heart_static.2019-12-09_02.29.34, ls_trf: heart_static.2019-12-09_02.29.34, meta: *meta}} # fish1
                - {tensors: {lf: heart_static.2019-12-09_02.35.49, ls_trf: heart_static.2019-12-09_02.35.49, meta: *meta}} # fish1
                - {tensors: {lf: heart_static.2019-12-09_02.42.03, ls_trf: heart_static.2019-12-09_02.42.03, meta: *meta}} # fish1
                - {tensors: {lf: heart_static.2019-12-09_02.48.24, ls_trf: heart_static.2019-12-09_02.48.24, meta: *meta}} # fish1
                - {tensors: {lf: heart_static.2019-12-09_02.54.46, ls_trf: heart_static.2019-12-09_02.54.46, meta: *meta}} # fish1
            - transform_sample: *sample_trfs
              augment_sample: *train_sample_prep
              filters: []
              datasets:
                - {tensors: {lf: heart_static.2019-12-08_06.35.52, ls_trf: heart_static.2019-12-08_06.35.52, meta: *meta}, indices: 1-} # fish5 val selected
                - {tensors: {lf: heart_static.2019-12-08_06.38.47, ls_trf: heart_static.2019-12-08_06.38.47, meta: *meta}, indices: 1-} # fish5 val selected
                - {tensors: {lf: heart_static.2019-12-08_06.10.34, ls_trf: heart_static.2019-12-08_06.10.34, meta: *meta}, indices: 1-} # fish5 val selected
                - {tensors: {lf: heart_static.2019-12-08_06.41.39, ls_trf: heart_static.2019-12-08_06.41.39, meta: *meta}, indices: 1-} # fish5 val selected
                - {tensors: {lf: heart_static.2019-12-08_06.18.09, ls_trf: heart_static.2019-12-08_06.18.09, meta: *meta}, indices: 1-} # fish5 val selected
                - {tensors: {lf: heart_static.2019-12-08_06.46.09, ls_trf: heart_static.2019-12-08_06.46.09, meta: *meta}, indices: 1-} # fish5 val selected
                - {tensors: {lf: heart_static.2019-12-08_06.23.13, ls_trf: heart_static.2019-12-08_06.23.13, meta: *meta}, indices: 1-} # fish5 val selected
                - {tensors: {lf: heart_static.2019-12-08_06.49.08, ls_trf: heart_static.2019-12-08_06.49.08, meta: *meta}, indices: 1-} # fish5 val selected
                - {tensors: {lf: heart_static.2019-12-08_06.25.02, ls_trf: heart_static.2019-12-08_06.25.02, meta: *meta}, indices: 1-} # fish5 val selected
                - {tensors: {lf: heart_static.2019-12-08_06.51.57, ls_trf: heart_static.2019-12-08_06.51.57, meta: *meta}, indices: 1-} # fish5 val selected
                - {tensors: {lf: heart_static.2019-12-08_06.30.40, ls_trf: heart_static.2019-12-08_06.30.40, meta: *meta}, indices: 1-} # fish5 val selected
        """
    else:
        raise NotImplementedError(name)

    return ConcatDataset(sections)
