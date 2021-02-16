import collections
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy
import torch.utils.data

from hylfm.datasets import ConcatDataset, TensorInfo, ZipDataset, get_dataset_from_info, get_tensor_info
from hylfm.hylfm_types import DatasetChoice, DatasetPart, TransformLike, TransformsPipeline
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


def get_dataset(name: DatasetChoice, part: DatasetPart, transforms_pipeline: TransformsPipeline):
    sliced = name.value.endswith("_sliced") and part == DatasetPart.train

    # sections will not be sampled across, which allows differentely sized images in the same dataset
    # subsections are grouped together to form mini-batches, thus their size needs to match
    sections: List[List[torch.utils.data.Dataset]] = []

    if name == DatasetChoice.beads_sample0:
        sections.append(
            [
                get_dataset_subsection(
                    tensors={
                        "lf": "local.beads.b01highc_0",
                        "ls_reg": "local.beads.b01highc_0",
                        "meta": transforms_pipeline.meta,
                    },
                    filters=[],
                    indices={DatasetPart.train: [0, 1, 2], DatasetPart.validate: [2, 3], DatasetPart.test: [3]}[part],
                    preprocess_sample=transforms_pipeline.sample_precache_trf,
                    augment_sample=transforms_pipeline.sample_preprocessing,
                )
            ]
        )
    elif name in [DatasetChoice.beads_highc_a, DatasetChoice.beads_highc_b]:
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
    elif name == DatasetChoice.heart_static_sample0:

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
        for tag in ["2019-12-08_06.35.52"]:  # fish5
            sections[-1].append(
                get_dataset_subsection(
                    tensors=get_tensors(tag),
                    filters=[],
                    indices=indices,
                    preprocess_sample=transforms_pipeline.sample_precache_trf,
                    augment_sample=transforms_pipeline.sample_preprocessing,
                )
            )

    elif (
        name in [DatasetChoice.heart_static_a, DatasetChoice.heart_static_c]
        or name
        in [
            DatasetChoice.heart_static_fish2,
            DatasetChoice.heart_static_fish2_sliced,
            DatasetChoice.heart_static_fish2_f4,
            DatasetChoice.heart_static_fish2_f4_sliced,
        ]
        and part != DatasetPart.test
    ):

        def get_tensors(tag_: str):
            return {
                "lf_repeat241" if sliced else "lf": f"heart_static.{tag_}",
                "ls_slice" if sliced else "ls_trf": f"heart_static.{tag_}",
                "meta": transforms_pipeline.meta,
            }

        if name.name.endswith("_sliced"):
            filters = [("z_range", {})]
            idx_first_vol = 209
        else:
            filters = []
            idx_first_vol = 1

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
                        filters=filters,
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
                        filters=filters,
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
                        filters=filters,
                        indices=slice(idx_first_vol, None, None),
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
                        filters=filters,
                        indices=[0],
                        preprocess_sample=transforms_pipeline.sample_precache_trf,
                        augment_sample=transforms_pipeline.sample_preprocessing,
                    )
                )

        elif part == DatasetPart.test and not name in (
            DatasetChoice.heart_static_fish2,
            DatasetChoice.heart_static_fish2_sliced,
            DatasetChoice.heart_static_fish2_f4,
            DatasetChoice.heart_static_fish2_f4_sliced,
        ):
            sections.append([])
            for tag in [  # fish2
                "2019-12-09_08.15.07",
                "2019-12-09_08.19.40",
                "2019-12-09_08.27.14",
                "2019-12-09_08.34.44",
                "2019-12-09_08.41.41",
                "2019-12-09_08.51.01",
                "2019-12-09_09.01.28",
                "2019-12-09_09.11.59",
                "2019-12-09_09.18.01",
                "2019-12-09_09.52.38",
                # "2019-12-09_07.42.47",  # no lr or bad quality
                # "2019-12-09_07.50.24",  # no lr or bad quality
            ]:
                sections[-1].append(
                    get_dataset_subsection(
                        tensors=get_tensors(tag),
                        filters=filters,
                        indices=None,
                        preprocess_sample=transforms_pipeline.sample_precache_trf,
                        augment_sample=transforms_pipeline.sample_preprocessing,
                    )
                )

        else:
            raise NotImplementedError(part)

    elif name == DatasetChoice.heart_static_b:

        def get_tensors(tag_: str):
            return {"lf": f"heart_static.{tag_}", "ls_trf": f"heart_static.{tag_}", "meta": transforms_pipeline.meta}

        if part == DatasetPart.train:
            sections.append([])
            for tag in [
                # fish1: Heart_tightCrop
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
            for tag in [
                # fish2: staticHeartFOV
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
                        indices=slice(1, None, None),
                        preprocess_sample=transforms_pipeline.sample_precache_trf,
                        augment_sample=transforms_pipeline.sample_preprocessing,
                    )
                )

            sections.append([])
            for tag in [
                # fish3: staticHeartFOV
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
            for tag in [
                # fish2: staticHeartFOV
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
                        indices=[0],
                        preprocess_sample=transforms_pipeline.sample_precache_trf,
                        augment_sample=transforms_pipeline.sample_preprocessing,
                    )
                )

            sections.append([])
            for tag in [
                # fish3: staticHeartFOV
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

    elif name == DatasetChoice.heart_static_c_care_complex:
        raise ValueError("not in use")
        if part == DatasetPart.test:
            tensor_infos = {
                "lfd": TensorInfo(
                    name="lfd",
                    root=Path("/g/kreshuk/LF_computed/lnet/plain/heart/static1/test/lr"),
                    location="*.tif",
                    transforms=tuple(),
                    datasets_per_file=1,
                    samples_per_dataset=1,
                    remove_singleton_axes_at=tuple(),
                    insert_singleton_axes_at=(0, 0),
                    z_slice=None,
                    skip_indices=tuple(),
                    meta=None,
                ),
                "care": TensorInfo(
                    name="care",
                    root=Path("/g/kreshuk/LF_computed/lnet/plain/heart/static1/test/v0_on_48x88x88"),
                    location="*.tif",
                    transforms=tuple(),
                    datasets_per_file=1,
                    samples_per_dataset=1,
                    remove_singleton_axes_at=tuple(),
                    insert_singleton_axes_at=(0, 0),
                    z_slice=None,
                    skip_indices=tuple(),
                    meta=None,
                ),
            }
            datasets = {
                k: get_dataset_from_info(dsinfo, cache=True, filters=[], indices=None)
                for k, dsinfo in tensor_infos.items()
            }

            spim_dataset = torch.utils.data.ConcatDataset(
                [
                    get_dataset_subsection(
                        tensors={
                            "lf": f"heart_static.{tag}",
                            "ls_trf": f"heart_static.{tag}",
                            "meta": transforms_pipeline.meta,
                        },
                        filters=[],
                        indices=None,
                        preprocess_sample=transforms_pipeline.sample_precache_trf,
                        augment_sample=transforms_pipeline.sample_preprocessing,
                    )
                    for tag in [  # fish2
                        "2019-12-09_08.15.07",
                        "2019-12-09_08.19.40",
                        "2019-12-09_08.27.14",
                        "2019-12-09_08.34.44",
                        "2019-12-09_08.41.41",
                        "2019-12-09_08.51.01",
                        "2019-12-09_09.01.28",
                        "2019-12-09_09.11.59",
                        "2019-12-09_09.18.01",
                        "2019-12-09_09.52.38",
                        # "2019-12-09_07.42.47",  # no lr or bad quality
                        # "2019-12-09_07.50.24",  # no lr or bad quality
                    ]
                ]
            )
            datasets["ls_trf"] = spim_dataset

            static1_dataset = ZipDataset(**datasets, transform=transforms_pipeline.sample_preprocessing)

            sections.append([static1_dataset])
        else:
            raise NotImplementedError("see commit ed5c7b02eaaada4fea244f5727f3ea7f0acb3459")

    elif (
        name
        in [
            DatasetChoice.heart_static_fish2,
            DatasetChoice.heart_static_fish2_f4,
            DatasetChoice.heart_static_fish2_sliced,
            DatasetChoice.heart_static_fish2_f4_sliced,
        ]
        and part == DatasetPart.test
    ):
        if "_f4" in name.name:
            skip_indices = []
        else:
            skip_indices = numpy.arange(100, 166).tolist()

        tensor_infos = {
            tensor_name: TensorInfo(
                name=tensor_name,
                root=Path("/g/kreshuk/beuttenm/hylfm-datasets/heart_static_fish2_f4") / tensor_name,
                location="*.tif",
                transforms=tuple(),
                datasets_per_file=1,
                samples_per_dataset=1,
                remove_singleton_axes_at=(-1,) if tensor_name in ("lf", "spim") else tuple(),
                insert_singleton_axes_at=(0, 0),
                z_slice=None,
                skip_indices=skip_indices,
                meta=None,
            )
            for tensor_name in ("lf", "spim", "care", "lfd")
        }
        # if name.name.endswith("_sliced"):  # saved already is filtered
        #     filters = [("z_range", {})]
        # else:
        filters = []

        datasets = {
            k: get_dataset_from_info(dsinfo, cache=True, filters=filters, indices=None)
            for k, dsinfo in tensor_infos.items()
        }

        heart_static_fish2_dataset = ZipDataset(datasets, transform=transforms_pipeline.sample_preprocessing)
        sections.append([heart_static_fish2_dataset])
    elif name in [DatasetChoice.heart_dyn_refine, DatasetChoice.heart_dyn_refine_lfd]:
        filters = [("z_range", {})]
        split_at = 964
        if part == DatasetPart.train:
            indices = slice(split_at, None, None)
        elif part == DatasetPart.validate:
            indices = slice(0, split_at, 4)
        elif part == DatasetPart.test:
            indices = slice(0, split_at)
        else:
            raise NotImplementedError(part)

        tag = "2019-12-09_04.54.38"
        if name == DatasetChoice.heart_dyn_refine:
            sections.append(
                [
                    get_dataset_subsection(
                        tensors={
                            "lf": f"heart_dynamic.{tag}",
                            "ls_slice": f"heart_dynamic.{tag}",
                            "meta": transforms_pipeline.meta,
                        },
                        filters=filters,
                        indices=indices,
                        preprocess_sample=transforms_pipeline.sample_precache_trf,
                        augment_sample=transforms_pipeline.sample_preprocessing,
                    )
                ]
            )
        elif name == DatasetChoice.heart_dyn_refine_lfd:
            # ls_slice_pre_cache_trf = [
            #     trf
            #     for trf in transforms_pipeline.sample_precache_trf
            #     if any([kwargs["apply_to"] == "ls_slice" for kwargs in trf.values()])
            # ]

            def ds_from_path(tensor_name: str, path: Path):
                pre_cache_trf = [
                    trf
                    for trf in transforms_pipeline.sample_precache_trf
                    if any([kwargs["apply_to"] == tensor_name for kwargs in trf.values()])
                ]
                tensor_info = TensorInfo(
                    name=tensor_name,
                    root=path,
                    location="*.tif",
                    transforms=pre_cache_trf,
                    datasets_per_file=1,
                    samples_per_dataset=1,
                    remove_singleton_axes_at=(-1,) if tensor_name in ("lf", "spim") else tuple(),
                    insert_singleton_axes_at=(0, 0),
                    z_slice=None,
                    skip_indices=skip_indices,
                    meta=None,
                )
                return get_dataset_from_info(
                    tensor_info, transforms=pre_cache_trf, cache=True, filters=[], indices=slice(4 * 209, None, None)
                )

            zipped_ds = ZipDataset(
                dict(
                    # ls_slice=get_dataset_from_info(
                    #     get_tensor_info(
                    #         info_name=f"heart_dynamic.{tag}", name="ls_slice", meta=transforms_pipeline.meta
                    #     ),
                    #     transforms=ls_slice_pre_cache_trf,
                    #     cache=True,
                    #     filters=filters,
                    #     indices=indices,
                    # ),
                    ls_slice=ds_from_path(
                        "ls_slice", Path("/g/kreshuk/LF_computed/lnet/plain/heart/dynamic1/test/ls_slice")
                    ),
                    lfd=ds_from_path("lfd", Path("/g/kreshuk/LF_computed/lnet/plain/heart/dynamic1/test/lr")),
                    care=ds_from_path(
                        "care", Path("/g/kreshuk/LF_computed/lnet/plain/heart/dynamic1/test/v0_on_48x88x88/")
                    ),
                ),
                transform=transforms_pipeline.sample_preprocessing,
            )
            sections.append([zipped_ds])

    elif name in [
        DatasetChoice.heart_2020_02_fish1_static,
        DatasetChoice.heart_2020_02_fish1_static_sliced,
        DatasetChoice.heart_2020_02_fish2_static,
        DatasetChoice.heart_2020_02_fish2_static_sliced,
    ]:

        def get_tensors(tag_: str):
            return {
                "lf_repeat241" if sliced else "lf": f"heart_static.{tag_}",
                "ls_slice" if sliced else "ls_trf": f"heart_static.{tag_}",
                "meta": transforms_pipeline.meta,
            }

        if name.name.endswith("_sliced"):
            filters = [("z_range", {})]
            val_until = 209 * 4
        else:
            filters = []
            val_until = 4

        if "fish1" in name.name:
            tag = "heart_2020_02_fish1_static"
        elif "fish2" in name.name:
            tag = "heart_2020_02_fish2_static"
        else:
            raise NotImplementedError(name)

        if part == DatasetPart.train:
            indices = slice(val_until, None, None)
        elif part == DatasetPart.validate:
            indices = slice(None, val_until, None)
        elif part == DatasetPart.test:
            indices = None
        else:
            raise NotImplementedError(part)

        sections.append(
            [
                get_dataset_subsection(
                    tensors=get_tensors(tag),
                    filters=filters,
                    indices=indices,
                    preprocess_sample=transforms_pipeline.sample_precache_trf,
                    augment_sample=transforms_pipeline.sample_preprocessing,
                )
            ]
        )
    else:
        raise NotImplementedError(name)

    return ConcatDataset([torch.utils.data.ConcatDataset(subsections) for subsections in sections])
