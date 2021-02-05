import collections
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy
import torch.utils.data

from hylfm.datasets import ConcatDataset, TensorInfo, ZipDataset, get_dataset_from_info, get_tensor_info
from hylfm.hylfm_types import DatasetName, DatasetPart, TransformLike, TransformsPipeline
from hylfm.transforms import Identity


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
        info.transforms += trfs_for_name
        if "_repeat" in name:
            name, _ = name.split("_repeat")

        infos[name] = info

    if isinstance(indices, list):
        assert all(isinstance(i, int) for i in indices)
    elif isinstance(indices, int):
        indices = [indices]
    elif indices is None:
        indices = slice(None)
    elif isinstance(indices, slice):
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
        transform=augment_sample,
    )


def get_dataset(name: DatasetName, part: DatasetPart, transforms_pipeline: TransformsPipeline):
    sections = []
    if name == DatasetName.beads_sample0:
        sections.append(
            [
                get_dataset_subsection(
                    tensors={
                        "lf": "local.beads.b01highc_0",
                        "ls_reg": "local.beads.b01highc_0",
                        "meta": transforms_pipeline.meta,
                    },
                    filters=[],
                    indices={DatasetPart.train: [0], DatasetPart.validate: [1], DatasetPart.test: [2]}[part],
                    preprocess_sample=transforms_pipeline.sample_precache_trf,
                    augment_sample=transforms_pipeline.sample_preprocessing,
                )
            ]
        )
    elif name in [DatasetName.beads_highc_a, DatasetName.beads_highc_b]:
        if part == DatasetPart.train:
            tensors = {"lf": f"beads.small_2", "ls_reg": f"beads.small_2", "meta": transforms_pipeline.meta}
        elif part == DatasetPart.validate:
            tensors = {"lf": f"beads.small_0", "ls_reg": f"beads.small_0", "meta": transforms_pipeline.meta}
        elif part == DatasetPart.test:
            tensors = {"lf": f"beads.small_1", "ls_reg": f"beads.small_1", "meta": transforms_pipeline.meta}
        else:
            raise NotImplementedError(part)

        sections.append(
            [
                get_dataset_subsection(
                    tensors=tensors,
                    filters=[],
                    indices=None,
                    preprocess_sample=transforms_pipeline.sample_precache_trf,
                    augment_sample=transforms_pipeline.sample_preprocessing,
                )
            ]
        )
    elif name == DatasetName.heart_static_sample0:

        def get_tensors(tag_: str):
            return {"lf": f"heart_static.{tag_}", "ls_trf": f"heart_static.{tag_}", "meta": transforms_pipeline.meta}

        if part == DatasetPart.train:
            indices = [0, 1]
        elif part == DatasetPart.validate:
            indices = [2]
        elif part == DatasetPart.test:
            indices = [3, 4]
        else:
            raise NotImplementedError(part)

        sections.append([])
        for tag in [  # fish5
            "2019-12-08_06.35.52",
        ]:
            sections[-1].append(
                get_dataset_subsection(
                    tensors=get_tensors(tag),
                    filters=[],
                    indices=indices,
                    preprocess_sample=transforms_pipeline.sample_precache_trf,
                    augment_sample=transforms_pipeline.sample_preprocessing,
                )
            )

    elif name in [DatasetName.heart_static_a, DatasetName.heart_static_c]:

        def get_tensors(tag_: str):
            return {"lf": f"heart_static.{tag_}", "ls_trf": f"heart_static.{tag_}", "meta": transforms_pipeline.meta}

        if part == DatasetPart.train:
            sections.append([])
            for tag in [  # fish3
                "2019-12-10_04.24.29",
                "2019-12-10_05.14.57",
                "2019-12-10_05.41.48",
                "2019-12-10_06.03.37",
                "2019-12-10_06.25.14",
            ]:
                sections[-1].append(
                    get_dataset_subsection(
                        tensors=get_tensors(tag),
                        filters=[],
                        indices=None,
                        preprocess_sample=transforms_pipeline.sample_precache_trf,
                        augment_sample=transforms_pipeline.sample_preprocessing,
                    )
                )

            sections.append([])
            for tag in [  # fish1
                "2019-12-09_02.16.30",
                "2019-12-09_02.23.01",
                "2019-12-09_02.29.34",
                "2019-12-09_02.35.49",
                "2019-12-09_02.42.03",
                "2019-12-09_02.48.24",
                "2019-12-09_02.54.46",
            ]:
                sections[-1].append(
                    get_dataset_subsection(
                        tensors=get_tensors(tag),
                        filters=[],
                        indices=None,
                        preprocess_sample=transforms_pipeline.sample_precache_trf,
                        augment_sample=transforms_pipeline.sample_preprocessing,
                    )
                )

            sections.append([])
            for tag in [  # fish5
                "2019-12-08_06.35.52",
                "2019-12-08_06.38.47",
                "2019-12-08_06.10.34",
                "2019-12-08_06.41.39",
                "2019-12-08_06.18.09",
                "2019-12-08_06.46.09",
                # "2019-12-08_06.23.13", len=1
                "2019-12-08_06.49.08",
                "2019-12-08_06.25.02",
                "2019-12-08_06.51.57",
                "2019-12-08_06.30.40",
            ]:
                sections[-1].append(
                    get_dataset_subsection(
                        tensors=get_tensors(tag),
                        filters=[],
                        indices=slice(1, None, None),
                        preprocess_sample=transforms_pipeline.sample_precache_trf,
                        augment_sample=transforms_pipeline.sample_preprocessing,
                    )
                )

        elif part == DatasetPart.validate:
            sections.append([])
            for tag in [  # fish5
                "2019-12-08_06.35.52",
                "2019-12-08_06.38.47",
                "2019-12-08_06.10.34",
                "2019-12-08_06.41.39",
                "2019-12-08_06.18.09",
                "2019-12-08_06.46.09",
                # "2019-12-08_06.23.13", len=1
                "2019-12-08_06.49.08",
                "2019-12-08_06.25.02",
                "2019-12-08_06.51.57",
                "2019-12-08_06.30.40",
            ]:
                sections[-1].append(
                    get_dataset_subsection(
                        tensors=get_tensors(tag),
                        filters=[],
                        indices=[0],
                        preprocess_sample=transforms_pipeline.sample_precache_trf,
                        augment_sample=transforms_pipeline.sample_preprocessing,
                    )
                )

        elif part == DatasetPart.test:
            sections.append([])
            for tag in [  # fish2
                "2019-12-09_09.52.38",
                "2019-12-09_08.34.44",
                "2019-12-09_08.41.41",
                "2019-12-09_08.51.01",
                "2019-12-09_09.01.28",
                "2019-12-09_09.11.59",
                "2019-12-09_09.18.01",
                "2019-12-09_08.15.07",
                "2019-12-09_08.19.40",
                "2019-12-09_08.27.14",
                "2019-12-09_07.42.47",
                "2019-12-09_07.50.24",
            ]:
                sections[-1].append(
                    get_dataset_subsection(
                        tensors=get_tensors(tag),
                        filters=[],
                        indices=None,
                        preprocess_sample=transforms_pipeline.sample_precache_trf,
                        augment_sample=transforms_pipeline.sample_preprocessing,
                    )
                )

        else:
            raise NotImplementedError(part)

    elif name == DatasetName.heart_static_b:

        def get_tensors(tag_: str):
            return {"lf": f"heart_static.{tag_}", "ls_trf": f"heart_static.{tag_}", "meta": transforms_pipeline.meta}

        if part == DatasetPart.train:
            sections.append([])
            for tag in [  # fish1: Heart_tightCrop
                "2019-12-09_02.16.30",
                "2019-12-09_02.23.01",
                "2019-12-09_02.29.34",
                "2019-12-09_02.35.49",
                "2019-12-09_02.42.03",
                "2019-12-09_02.48.24",
                "2019-12-09_02.54.46",
            ]:
                sections[-1].append(
                    get_dataset_subsection(
                        tensors=get_tensors(tag),
                        filters=[],
                        indices=slice(1, None, None),
                        preprocess_sample=transforms_pipeline.sample_precache_trf,
                        augment_sample=transforms_pipeline.sample_preprocessing,
                    )
                )

            sections.append([])
            for tag in [  # fish2, fish3: staticHeartFOV
                # fish2
                "2019-12-09_09.52.38",
                "2019-12-09_08.34.44",
                "2019-12-09_08.41.41",
                "2019-12-09_08.51.01",
                "2019-12-09_09.01.28",
                "2019-12-09_09.11.59",
                "2019-12-09_09.18.01",
                "2019-12-09_08.15.07",
                "2019-12-09_08.19.40",
                "2019-12-09_08.27.14",
                "2019-12-09_07.42.47",
                "2019-12-09_07.50.24",
                # fish3
                "2019-12-10_04.24.29",
                "2019-12-10_05.14.57",
                "2019-12-10_05.41.48",
                "2019-12-10_06.03.37",
                "2019-12-10_06.25.14",
            ]:

                sections[-1].append(
                    get_dataset_subsection(
                        tensors=get_tensors(tag),
                        filters=[],
                        indices=slice(1, None, None),
                        preprocess_sample=transforms_pipeline.sample_precache_trf,
                        augment_sample=transforms_pipeline.sample_preprocessing,
                    )
                )

        elif part == DatasetPart.validate:
            sections.append([])
            for tag in [  # fish1: Heart_tightCrop
                "2019-12-09_02.16.30",
                "2019-12-09_02.23.01",
                "2019-12-09_02.29.34",
                "2019-12-09_02.35.49",
                "2019-12-09_02.42.03",
                "2019-12-09_02.48.24",
                "2019-12-09_02.54.46",
            ]:
                sections[-1].append(
                    get_dataset_subsection(
                        tensors=get_tensors(tag),
                        filters=[],
                        indices=[0],
                        preprocess_sample=transforms_pipeline.sample_precache_trf,
                        augment_sample=transforms_pipeline.sample_preprocessing,
                    )
                )

            sections.append([])
            for tag in [  # fish2, fish3: staticHeartFOV
                # fish2
                "2019-12-09_09.52.38",
                "2019-12-09_08.34.44",
                "2019-12-09_08.41.41",
                "2019-12-09_08.51.01",
                "2019-12-09_09.01.28",
                "2019-12-09_09.11.59",
                "2019-12-09_09.18.01",
                "2019-12-09_08.15.07",
                "2019-12-09_08.19.40",
                "2019-12-09_08.27.14",
                "2019-12-09_07.42.47",
                "2019-12-09_07.50.24",
                # fish3
                "2019-12-10_04.24.29",
                "2019-12-10_05.14.57",
                "2019-12-10_05.41.48",
                "2019-12-10_06.03.37",
                "2019-12-10_06.25.14",
            ]:

                sections[-1].append(
                    get_dataset_subsection(
                        tensors=get_tensors(tag),
                        filters=[],
                        indices=[0],
                        preprocess_sample=transforms_pipeline.sample_precache_trf,
                        augment_sample=transforms_pipeline.sample_preprocessing,
                    )
                )

        elif part == DatasetPart.test:
            sections.append([])
            for tag in [  # fish5
                "2019-12-08_06.35.52",
                "2019-12-08_06.38.47",
                "2019-12-08_06.10.34",
                "2019-12-08_06.41.39",
                "2019-12-08_06.18.09",
                "2019-12-08_06.46.09",
                "2019-12-08_06.23.13",  # len=1
                "2019-12-08_06.49.08",
                "2019-12-08_06.25.02",
                "2019-12-08_06.51.57",
                "2019-12-08_06.30.40",
            ]:
                sections[-1].append(
                    get_dataset_subsection(
                        tensors=get_tensors(tag),
                        filters=[],
                        indices=None,
                        preprocess_sample=transforms_pipeline.sample_precache_trf,
                        augment_sample=transforms_pipeline.sample_preprocessing,
                    )
                )

        else:
            raise NotImplementedError(part)
    else:
        raise NotImplementedError(name)

    return ConcatDataset([torch.utils.data.ConcatDataset(subsections) for subsections in sections])
