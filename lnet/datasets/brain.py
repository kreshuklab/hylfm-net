import argparse
import re
from pathlib import Path

import yaml

from lnet.datasets.base import TensorInfo, get_dataset_from_info
from lnet.settings import settings
from lnet.transformations.affine_utils import get_lf_shape, get_ls_shape, get_raw_ls_crop


def get_gcamp_z_slice_from_tag2(tag2: str):
    regular_exp_LS_pos = "SinglePlane_(-[0-9]{3})"
    LS_stack_range_start = -450
    # LS_stack_range_end = -210
    return int(re.search(regular_exp_LS_pos, tag2).group(1)) - LS_stack_range_start


def get_tensor_info(tag: str, name: str, meta: dict):
    meta = dict(meta)
    assert "nnum" in meta
    assert "interpolation_order" in meta
    assert "scale" in meta
    meta[
        "z_ls_rescaled"
    ] = 241  # all gcamp data z_slices are conidered as one of 241 slices, 1mu apart, in a 241um volume

    root = "GKRESHUK"
    insert_singleton_axes_at = [0, 0]
    if "_repeat" in name:
        name, repeat = name.split("_repeat")
        repeat = int(repeat)
    else:
        repeat = 1

    # data quality:
    # 4 amazing
    # 3 very good
    # 2  good
    # 1   blurry
    meta["quality"] = 2

    # neuron activity in data:
    # 4   amazing
    # 3 very much
    # 2      some
    # 1   nothing
    meta["activity"] = 1

    fish, tag1, tag2 = tag.split("__")

    if name == "ls_slice":
        if "SinglePlane" in tag2:
            z_slice = get_gcamp_z_slice_from_tag2(tag2)
        elif "SwipeThrough" in tag2:
            _, lower, upper, _, samples_per_dataset = tag2.split("_")
            lower = int(lower)
            upper = int(upper)
            samples_per_dataset = int(samples_per_dataset)
            step_float = abs(upper - lower) / (samples_per_dataset - 1)
            step = int(step_float)
            assert step == step_float, (step, step_float)
            offset = 450 + lower
            assert offset == -210 - upper, (upper, lower)
            z_slice = f"{offset}+idx%{samples_per_dataset}*{step}"
        else:
            raise NotImplementedError((tag2, tag))
    else:
        z_slice = None

    tag = f"{tag1}__{tag2}"
    if tag1.endswith("_short"):
        tag1 = tag1[: -len("_short")]
        short = True
    else:
        short = False

    if fish == "beads_after_fish":
        location = (
            f"LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish2/beads_after_fish/{tag1}/"
        )

    elif fish == "08_1":
        meta["quality"] = 1
        location = f"LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200308_Gcamp/brain/{tag1}/"
        if name == "ls_slice" and "SinglePlane" in tag2:
            if tag1 == "2020-03-08_06.00.27":
                samples_per_dataset = 250
            elif tag1 == "2020-03-08_06.06.38":
                samples_per_dataset = 250
            elif tag1 == "2020-03-08_06.26.41":
                samples_per_dataset = 300
            elif tag1 == "2020-03-08_06.26.41":
                samples_per_dataset = 300
            elif tag1 == "2020-03-08_06.38.34":
                samples_per_dataset = 300
            elif tag1 == "2020-03-08_06.38.34":
                samples_per_dataset = 300
            elif tag1 == "2020-03-08_06.38.34":
                samples_per_dataset = 300
            elif tag1 == "2020-03-08_06.38.34":
                samples_per_dataset = 300
            elif tag1 == "2020-03-08_06.38.34":
                samples_per_dataset = 300
            elif tag1 == "2020-03-08_06.38.34":
                samples_per_dataset = 300
            elif tag1 == "2020-03-08_06.38.34":
                samples_per_dataset = 300
            elif tag1 == "2020-03-08_06.38.34":
                samples_per_dataset = 300
            elif tag1 == "2020-03-08_06.38.34":
                samples_per_dataset = 300
            else:
                raise NotImplementedError((tag1, tag))

    elif fish == "09_1" and tag1 == "2020-03-09_04.35.55":
        location = f"LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200309_Gcamp/fish1_awesome/{tag1}/"
        if name == "ls_slice" and "SinglePlane" in tag2:
            samples_per_dataset = 600

    elif fish == "09_1" and tag1 == "2020-03-09_02.53.02":
        location = (
            f"LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200309_Gcamp/savedOnSSD/fish1/{tag1}/"
        )
        if name == "ls_slice" and "SinglePlane" in tag2:
            samples_per_dataset = 600

    elif fish == "09_2":
        location = (
            f"LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200309_Gcamp/fish2_kinda_crap/{tag1}/"
        )

    elif fish == "09_3":
        location = f"LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200309_Gcamp/fish3/{tag1}/"
        if name == "ls_slice" and "SinglePlane" in tag2:
            samples_per_dataset = 600

    elif fish == "09_4":
        location = f"LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200309_Gcamp/fish4_promising/"
        if tag1 in [
            "2020-03-09_07.14.47",
            "2020-03-09_07.14.47",
            "2020-03-09_07.26.02",
            "2020-03-09_07.26.02",
            "2020-03-09_07.30.14",
            "2020-03-09_07.50.51",
            "2020-03-09_07.51.52",
        ]:
            pass
        elif tag1 in [
            "2020-03-09_08.20.04",
            "2020-03-09_08.20.04",
            "2020-03-09_08.21.40",
            "2020-03-09_08.21.40",
            "2020-03-09_08.31.10",
        ]:
            location += "longRun/fov1/"
        elif tag1 in ["2020-03-09_08.53.20", "2020-03-09_09.06.55"]:
            location += "longRun/fov2/"
        elif tag1 in [
            "2020-03-09_08.41.22",
            "2020-03-09_08.41.22",
            "2020-03-09_08.41.22",
            "2020-03-09_08.41.22",
            "2020-03-09_08.41.22",
            "2020-03-09_08.41.22",
            "2020-03-09_08.41.22",
        ]:
            location += "singlePlanes/fov1/"
        elif tag1 in [
            "2020-03-09_08.56.54",
            "2020-03-09_09.31.52",
            "2020-03-09_09.31.52",
            "2020-03-09_09.31.52",
            "2020-03-09_09.31.52",
            "2020-03-09_09.31.52",
            "2020-03-09_09.31.52",
            "2020-03-09_09.31.52",
            "2020-03-09_09.31.52",
            "2020-03-09_09.31.52",
            "2020-03-09_09.31.52",
            "2020-03-09_09.31.52",
            "2020-03-09_09.31.52",
            "2020-03-09_09.31.52",
            "2020-03-09_09.31.52",
            "2020-03-09_09.31.52",
            "2020-03-09_09.31.52",
            "2020-03-09_09.31.52",
            "2020-03-09_09.31.52",
            "2020-03-09_09.31.52",
            "2020-03-09_09.31.52",
            "2020-03-09_09.31.52",
            "2020-03-09_09.31.52",
            "2020-03-09_09.31.52",
            "2020-03-09_09.31.52",
            "2020-03-09_09.31.52",
        ]:
            location += "singlePlanes/fov2/"
        elif tag1 in "2020-03-09_07.53.40":
            location += "superFast_50Hz/"
        else:
            raise NotImplementedError((tag1, tag))

        location += f"{tag1}/"
        if name == "ls_slice" and "SinglePlane" in tag2:
            samples_per_dataset = 600
    elif fish == "11_1":
        location = f"LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish1/5Hz/"
        if tag1 in [
            "2020-03-11_02.34.23",
            "2020-03-11_02.48.29",
            "2020-03-11_02.49.27",
            "2020-03-11_02.54.20",
            "2020-03-11_03.22.33",
            "2020-03-11_06.16.29",
            "2020-03-11_02.43.33",
        ]:
            location += "longRun/"
        elif tag1 in ["2020-03-11_03.30.35", "2020-03-11_03.43.42", "2020-03-11_03.48.47", "2020-03-11_04.00.37"]:
            location += "longRun/niceOne/"
        elif tag1 in ["2020-03-11_04.03.27", "2020-03-11_04.17.17"]:
            location += "longRun/niceOne2/"
        elif tag1 in ["2020-03-11_04.25.22", "2020-03-11_04.27.34", "2020-03-11_04.28.39", "2020-03-11_04.35.02"]:
            location += "longRun/niceOne3/"
        elif tag1 == "2020-03-11_04.47.24":
            location += "longRun/niceOne4/"
        elif tag1 == "2020-03-11_05.53.13":
            location += "longRun/niceOne5/"
        elif tag1 in ["2020-03-11_05.26.26", "2020-03-11_05.36.29"]:
            location += "longRun/slideThrough/"
        elif tag1 == "2020-03-11_06.07.48":
            location += "singlePlanes/"
        else:
            raise NotImplementedError((tag1, tag))

        location += f"{tag1}/"
        if name == "ls_slice" and "SinglePlane" in tag2:
            if tag1 == "2020-03-11_02.34.23":
                samples_per_dataset = 300
            elif tag1 == "2020-03-11_03.22.33":
                samples_per_dataset = 150
            elif tag1 == "2020-03-11_02.00.31":
                samples_per_dataset = 600
            elif tag1 == "2020-03-11_04.25.22":
                samples_per_dataset = 150
            elif tag1 == "2020-03-11_03.43.42":
                samples_per_dataset = 150
            elif tag1 == "2020-03-11_03.48.47":
                samples_per_dataset = 150
            elif tag1 == "2020-03-11_04.00.37":
                samples_per_dataset = 150
            elif tag1 == "2020-03-11_06.07.48":
                samples_per_dataset = 600
            elif tag1 == "2020-03-11_06.07.48":
                samples_per_dataset = 600
            else:
                raise NotImplementedError((tag1, tag))

    elif fish == "11_2":
        location = f"LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish2/"
        if tag1 == "2020-03-11_07.34.47":
            location += "10Hz/"
        elif tag1 in ["2020-03-11_08.52.33", "2020-03-11_09.08.00"]:
            location += "10Hz/241Planes/"
        elif tag1 in [
            "2020-03-11_07.30.39",
            "2020-03-11_07.30.39",
            "2020-03-11_10.13.20",
            "2020-03-11_10.13.20",
            "2020-03-11_10.25.41",
            "2020-03-11_10.25.41",
            "2020-03-11_10.25.41",
        ]:
            location += "10Hz/singlePlane/"
        elif tag1 in ["2020-03-11_10.17.34", "2020-03-11_10.17.34", "2020-03-11_10.21.14", "2020-03-11_10.21.14"]:
            location += "10Hz/singlePlane/superNice/"
        elif tag1 in [
            "2020-03-11_08.30.21",
            "2020-03-11_08.34.19",
            "2020-03-11_08.36.58",
            "2020-03-11_08.39.31",
            "2020-03-11_08.42.07",
            "2020-03-11_08.44.50",
            "2020-03-11_08.47.29",
        ]:
            location += "10Hz/slideThrough/"
        elif tag1 in ["2020-03-11_10.33.17", "2020-03-11_10.34.15", "2020-03-11_10.35.41"]:
            location += "16ms/"
        elif tag1 in ["2020-03-11_08.19.37", "2020-03-11_08.20.29", "2020-03-11_08.22.19"]:
            location += "20Hz/"
        elif tag1 == "2020-03-11_08.12.13":
            location += "50Hz/"
        elif tag1 in ["2020-03-11_06.53.14", "2020-03-11_06.55.38", "2020-03-11_06.57.26"]:
            location += "5Hz/"
        elif tag1 in ["2020-03-11_10.06.25"]:
            location += "Heinbrain/"
        elif tag1 in ["2020-03-11_09.35.16", "2020-03-11_09.36.25", "2020-03-11_09.47.38", "2020-03-11_09.54.35"]:
            location += "opticTectum/10Hz/"

        location += f"/{tag1}/"
        if name == "ls_slice" and "SinglePlane" in tag2:
            if tag1 == "2020-03-11_07.30.39":
                samples_per_dataset = 600
            elif tag1 == "2020-03-11_07.30.39":
                samples_per_dataset = 600
            elif tag1 == "2020-03-11_10.13.20":
                samples_per_dataset = 600
            elif tag1 == "2020-03-11_10.13.20":
                samples_per_dataset = 600
            elif tag1 == "2020-03-11_10.25.41":
                samples_per_dataset = 600
            elif tag1 == "2020-03-11_10.25.41":
                samples_per_dataset = 600
            elif tag1 == "2020-03-11_10.25.41":
                samples_per_dataset = 600
            elif tag1 == "2020-03-11_10.17.34":
                samples_per_dataset = 600
            elif tag1 == "2020-03-11_10.17.34":
                samples_per_dataset = 600
            elif tag1 == "2020-03-11_10.21.14":
                samples_per_dataset = 600
            elif tag1 == "2020-03-11_10.21.14":
                samples_per_dataset = 600
            elif tag1 == "2020-03-11_10.33.17":
                samples_per_dataset = 200
            elif tag1 == "2020-03-11_10.34.15":
                samples_per_dataset = 400
            elif tag1 == "2020-03-11_10.35.41":
                samples_per_dataset = 1000
            elif tag1 == "2020-03-11_08.12.13":
                samples_per_dataset = 600
            elif tag1 == "2020-03-11_06.53.14":
                samples_per_dataset = 150
            elif tag1 == "2020-03-11_10.06.25":
                samples_per_dataset = 600
            elif tag1 == "2020-03-11_10.45.34":
                samples_per_dataset = 5
            else:
                raise NotImplementedError((tag1, tag))
    else:
        raise NotImplementedError(tag)

    # resolve stack_*_channel_* with tag2
    root_path = getattr(settings.data_roots, root)
    location_paths = list((root_path / location).glob(f"stack_*_channel_*/{tag2}/"))
    assert len(location_paths) == 1, location_paths
    location_path = location_paths[0]
    location = location_path.relative_to(root_path).parent.as_posix().strip("/") + "/"

    crop_name = "gcamp"
    if name == "lf":
        transformations = [{"Assert": {"apply_to": name, "expected_tensor_shape": [1, 1] + get_lf_shape(crop_name)}}]
        location += f"{tag2}/TP_{'00000' if short else '*'}/RC_rectified/Cam_Right_*_rectified.tif"
        samples_per_dataset = 1
    elif name == "ls_slice":
        location = location.replace("TestOutputGcamp/", "")
        transformations = [
            {"Assert": {"apply_to": name, "expected_tensor_shape": [1, 1, 1, 2048, 2060]}},  # raw ls shape
            {"FlipAxis": {"apply_to": name, "axis": 2}},
            {"Crop": {"apply_to": name, "crop": get_raw_ls_crop(crop_name, for_slice=True)}},
            {"Assert": {"apply_to": name, "expected_tensor_shape": [1, 1] + get_ls_shape(crop_name, for_slice=True)}},
            {
                "Resize": {
                    "apply_to": name,
                    "shape": [
                        1.0,
                        1.0,
                        meta.get("ls_slice_scale", meta["scale"]) / meta["nnum"],
                        meta.get("ls_slice_scale", meta["scale"]) / meta["nnum"],
                    ],
                    "order": meta["interpolation_order"],
                }
            },
            {"Cast": {"apply_to": name, "dtype": "float32", "device": "numpy"}},
            {
                "Assert": {
                    "apply_to": name,
                    "expected_tensor_shape": [None, 1, 1]
                    + [
                        s / meta["nnum"] * meta.get("ls_slice_scale", meta["scale"])
                        for s in get_ls_shape(crop_name, for_slice=True)[1:]
                    ],
                }
            },
        ]
        location += f"Cam_Left_{'00000' if short else '*'}.h5/Data"
    elif name == "lr":
        location = location.replace("LF_partially_restored/", "LF_computed/")
        location += f"{tag2}/TP_{'00000' if short else '*'}/RCout/Cam_Right_*.tif"
    else:
        raise NotImplementedError

    if location is None or location.endswith("/"):
        raise NotImplementedError(f"tag: {tag}, name: {name}")

    # assert tag.replace("_short", "") in location, (tag, name, location)
    if "crop_names" in meta:
        assert crop_name in meta["crop_names"]

    if "crop_name" in meta:
        assert meta["crop_name"] == crop_name, (meta["crop_name"], crop_name)
    else:
        meta["crop_name"] = crop_name

    return TensorInfo(
        name=name,
        root=root,
        location=location,
        insert_singleton_axes_at=insert_singleton_axes_at,
        transformations=transformations,
        z_slice=z_slice,
        samples_per_dataset=samples_per_dataset,
        repeat=repeat,
        tag=tag.replace("-", "").replace(".", ""),
        meta=meta,
    )


def check_filter(tag: str, meta: dict):
    lf_crops = {"gcamp": [[0, None], [0, None], [0, None]]}

    filters = [
        ("z_range", {"lf_crops": lf_crops}),
        ("signal2noise", {"apply_to": "ls_slice", "signal_percentile": 99.9, "noise_percentile": 5.0, "ratio": 1.5}),
    ]

    ds_unfiltered = get_dataset_from_info(get_tensor_info(tag, "ls_slice", meta=meta), cache=True)
    print(" unfiltered", len(ds_unfiltered))
    ds = get_dataset_from_info(get_tensor_info(tag, "ls_slice", meta=meta), cache=True, filters=filters)

    print("ds filtered", len(ds))


def check_data(tag: str, meta: dict):
    ls_slice = get_dataset_from_info(get_tensor_info(tag, "ls_slice", meta=meta), cache=True)
    if meta["scale"] == 4:
        lf = get_dataset_from_info(get_tensor_info(tag, "lf", meta=meta), cache=True)
        assert len(lf) == len(ls_slice), (tag, len(lf), len(ls_slice))

    assert len(ls_slice) > 0, tag
    print(tag, len(ls_slice))


def get_tags():
    with (Path(__file__).parent / "tags" / Path(__file__).with_suffix(".yml").name).open() as f:
        return [tag.strip() for tag in yaml.safe_load(f)]


def quick_check_all(meta: dict):
    for tag in get_tags():
        try:
            lf = get_dataset_from_info(get_tensor_info(tag, "lf", meta=meta), cache=False)
        except Exception as e:
            print(tag, e)
            lf = []
        try:
            ls_slice = get_dataset_from_info(get_tensor_info(tag, "ls_slice", meta=meta), cache=False)
        except Exception as e:
            print(tag, e)
            ls_slice = []

        if len(lf) != len(ls_slice) or len(ls_slice) == 0:
            print(tag, len(lf), len(ls_slice))
        else:
            print(tag, len(lf))

        # assert len(ls_slice) > 0, tag


if __name__ == "__main__":
    # try:
    parser = argparse.ArgumentParser()
    parser.add_argument("tagnr", type=int)
    parser.add_argument("meta_path", type=Path)

    args = parser.parse_args()

    tag = get_tags()[args.tagnr]
    with args.meta_path.open() as f:
        meta = yaml.safe_load(f)

    check_data(tag, meta=meta)
    check_filter(tag, meta=meta)
    # except:
    #     print("quick check")
    #     # meta = {"nnum": 19, "interpolation_order": 2, "pred_z_min": 152, "pred_z_max": 615, "shrink": 8, "scale": 8}
    #     # quick_check_all(meta=meta)

"""
single plane blinking, good for testing, traces:
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200308_Gcamp/brain/2020-03-08_06.06.38/stack_2_channel_3/SinglePlane_-330/TP_00001
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200309_Gcamp/fish1_awesome/2020-03-09_04.35.55/stack_10_channel_3/SinglePlane_-290/TP_00000

maybe:
LF_partially_restored/LenseLeNet_Microscope/20200311_Gcamp/fish1/5Hz/longRun/niceOne3/2020-03-11_04.25.22/stack_34_channel_4

=========================================
fish_08_1 	quality *
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200308_Gcamp/brain/2020-03-08_06.00.27/stack_2_channel_3/SinglePlane_-330
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200308_Gcamp/brain/2020-03-08_06.06.38/stack_2_channel_3/SinglePlane_-330
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200308_Gcamp/brain/2020-03-08_06.26.41/stack_1_channel_3/SwipeThrough_-450_-210_nimages_241
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200308_Gcamp/brain/2020-03-08_06.26.41/stack_2_channel_3/SinglePlane_-330
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200308_Gcamp/brain/2020-03-08_06.26.41/stack_3_channel_3/SinglePlane_-340
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200308_Gcamp/brain/2020-03-08_06.38.34/stack_10_channel_3/SinglePlane_-290
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200308_Gcamp/brain/2020-03-08_06.38.34/stack_1_channel_3/SwipeThrough_-450_-210_nimages_241
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200308_Gcamp/brain/2020-03-08_06.38.34/stack_2_channel_3/SinglePlane_-330
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200308_Gcamp/brain/2020-03-08_06.38.34/stack_3_channel_3/SinglePlane_-340
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200308_Gcamp/brain/2020-03-08_06.38.34/stack_4_channel_3/SinglePlane_-350
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200308_Gcamp/brain/2020-03-08_06.38.34/stack_5_channel_3/SinglePlane_-360
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200308_Gcamp/brain/2020-03-08_06.38.34/stack_6_channel_3/SinglePlane_-370
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200308_Gcamp/brain/2020-03-08_06.38.34/stack_7_channel_3/SinglePlane_-320
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200308_Gcamp/brain/2020-03-08_06.38.34/stack_8_channel_3/SinglePlane_-310
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200308_Gcamp/brain/2020-03-08_06.38.34/stack_9_channel_3/SinglePlane_-300
=========================================
fish_09_1, quality *** (double check), better for testing, as only single planes, traces possible
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200309_Gcamp/fish1_awesome/2020-03-09_04.35.55/stack_10_channel_3/SinglePlane_-290
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200309_Gcamp/fish1_awesome/2020-03-09_04.35.55/stack_12_channel_3/SinglePlane_-390
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200309_Gcamp/fish1_awesome/2020-03-09_04.35.55/stack_13_channel_3/SinglePlane_-380
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200309_Gcamp/fish1_awesome/2020-03-09_04.35.55/stack_14_channel_3/SinglePlane_-280
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200309_Gcamp/fish1_awesome/2020-03-09_04.35.55/stack_15_channel_3/SinglePlane_-270
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200309_Gcamp/fish1_awesome/2020-03-09_04.35.55/stack_2_channel_3/SinglePlane_-330
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200309_Gcamp/fish1_awesome/2020-03-09_04.35.55/stack_3_channel_3/SinglePlane_-340
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200309_Gcamp/fish1_awesome/2020-03-09_04.35.55/stack_4_channel_3/SinglePlane_-350
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200309_Gcamp/fish1_awesome/2020-03-09_04.35.55/stack_5_channel_3/SinglePlane_-360
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200309_Gcamp/fish1_awesome/2020-03-09_04.35.55/stack_6_channel_3/SinglePlane_-370
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200309_Gcamp/fish1_awesome/2020-03-09_04.35.55/stack_7_channel_3/SinglePlane_-320
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200309_Gcamp/fish1_awesome/2020-03-09_04.35.55/stack_8_channel_3/SinglePlane_-310
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200309_Gcamp/fish1_awesome/2020-03-09_04.35.55/stack_9_channel_3/SinglePlane_-300
=========================================
fish_09_1
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200309_Gcamp/savedOnSSD/fish1/2020-03-09_02.53.02/stack_11_channel_3/SwipeThrough_-390_-300_nimages_10
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200309_Gcamp/savedOnSSD/fish1/2020-03-09_02.53.02/stack_12_channel_3/SinglePlane_-390
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200309_Gcamp/savedOnSSD/fish1/2020-03-09_02.53.02/stack_13_channel_3/SinglePlane_-380
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200309_Gcamp/savedOnSSD/fish1/2020-03-09_02.53.02/stack_1_channel_3/SwipeThrough_-450_-210_nimages_241
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200309_Gcamp/savedOnSSD/fish1/2020-03-09_02.53.02/stack_2_channel_3/SinglePlane_-330
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200309_Gcamp/savedOnSSD/fish1/2020-03-09_02.53.02/stack_3_channel_3/SinglePlane_-340
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200309_Gcamp/savedOnSSD/fish1/2020-03-09_02.53.02/stack_4_channel_3/SinglePlane_-350
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200309_Gcamp/savedOnSSD/fish1/2020-03-09_02.53.02/stack_5_channel_3/SinglePlane_-360
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200309_Gcamp/savedOnSSD/fish1/2020-03-09_02.53.02/stack_6_channel_3/SinglePlane_-370
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200309_Gcamp/savedOnSSD/fish1/2020-03-09_02.53.02/stack_7_channel_3/SinglePlane_-320
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200309_Gcamp/savedOnSSD/fish1/2020-03-09_02.53.02/stack_8_channel_3/SinglePlane_-310
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200309_Gcamp/savedOnSSD/fish1/2020-03-09_02.53.02/stack_9_channel_3/SinglePlane_-300
=========================================
fish_09_2, quality *** (fires like crazy)
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200309_Gcamp/fish2_kinda_crap/2020-03-09_06.05.11/stack_11_channel_3/SwipeThrough_-390_-270_nimages_13
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200309_Gcamp/fish2_kinda_crap/2020-03-09_06.05.11/stack_1_channel_3/SwipeThrough_-450_-210_nimages_241
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200309_Gcamp/fish2_kinda_crap/2020-03-09_06.15.03/stack_11_channel_3/SwipeThrough_-390_-270_nimages_13
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200309_Gcamp/fish2_kinda_crap/2020-03-09_06.15.03/stack_1_channel_3/SwipeThrough_-450_-210_nimages_241
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200309_Gcamp/fish2_kinda_crap/2020-03-09_06.17.56/stack_11_channel_3/SwipeThrough_-390_-270_nimages_13
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200309_Gcamp/fish2_kinda_crap/2020-03-09_06.17.56/stack_1_channel_3/SwipeThrough_-450_-210_nimages_241
=========================================
fish_09_3, quality ** (single neurons on a party, crazy crazy blinking, e.g. 2020-03-09_06.43.40/stack_2_channel_3/SinglePlane_-330/TP_00000/)
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200309_Gcamp/fish3/2020-03-09_06.29.20/stack_11_channel_3/SwipeThrough_-390_-270_nimages_13
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200309_Gcamp/fish3/2020-03-09_06.29.20/stack_1_channel_3/SwipeThrough_-450_-210_nimages_241
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200309_Gcamp/fish3/2020-03-09_06.38.47/stack_11_channel_3/SwipeThrough_-390_-270_nimages_13
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200309_Gcamp/fish3/2020-03-09_06.38.47/stack_1_channel_3/SwipeThrough_-450_-210_nimages_241
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200309_Gcamp/fish3/2020-03-09_06.43.40/stack_11_channel_3/SwipeThrough_-390_-270_nimages_13
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200309_Gcamp/fish3/2020-03-09_06.43.40/stack_1_channel_3/SwipeThrough_-450_-210_nimages_241
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200309_Gcamp/fish3/2020-03-09_06.43.40/stack_2_channel_3/SinglePlane_-330
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200309_Gcamp/fish3/2020-03-09_06.43.40/stack_3_channel_3/SinglePlane_-340
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200309_Gcamp/fish3/2020-03-09_06.43.40/stack_7_channel_3/SinglePlane_-320
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200309_Gcamp/fish3/2020-03-09_06.54.12/stack_11_channel_3/SwipeThrough_-390_-270_nimages_13
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200309_Gcamp/fish3/2020-03-09_06.54.12/stack_1_channel_3/SwipeThrough_-450_-210_nimages_241
=========================================
fish_09_4, not so much going on ...
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200309_Gcamp/fish4_promising/2020-03-09_07.14.47/stack_11_channel_3/SwipeThrough_-390_-270_nimages_13
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200309_Gcamp/fish4_promising/2020-03-09_07.14.47/stack_1_channel_3/SwipeThrough_-450_-210_nimages_241
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200309_Gcamp/fish4_promising/2020-03-09_07.26.02/stack_11_channel_4/SwipeThrough_-390_-270_nimages_13
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200309_Gcamp/fish4_promising/2020-03-09_07.26.02/stack_1_channel_4/SwipeThrough_-450_-210_nimages_241
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200309_Gcamp/fish4_promising/2020-03-09_07.30.14/stack_1_channel_4/SwipeThrough_-450_-210_nimages_241
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200309_Gcamp/fish4_promising/2020-03-09_07.50.51/stack_11_channel_3/SwipeThrough_-390_-270_nimages_13
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200309_Gcamp/fish4_promising/2020-03-09_07.51.52/stack_11_channel_3/SwipeThrough_-390_-270_nimages_13
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200309_Gcamp/fish4_promising/longRun/fov1/2020-03-09_08.20.04/stack_11_channel_3/SwipeThrough_-390_-270_nimages_25
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200309_Gcamp/fish4_promising/longRun/fov1/2020-03-09_08.20.04/stack_1_channel_4/SwipeThrough_-450_-210_nimages_241
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200309_Gcamp/fish4_promising/longRun/fov1/2020-03-09_08.21.40/stack_11_channel_3/SwipeThrough_-390_-270_nimages_25
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200309_Gcamp/fish4_promising/longRun/fov1/2020-03-09_08.21.40/stack_1_channel_4/SwipeThrough_-450_-210_nimages_241
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200309_Gcamp/fish4_promising/longRun/fov1/2020-03-09_08.31.10/stack_28_channel_4/SwipeThrough_-450_-210_nimages_49
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200309_Gcamp/fish4_promising/longRun/fov2/2020-03-09_08.53.20/stack_28_channel_4/SwipeThrough_-450_-210_nimages_49
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200309_Gcamp/fish4_promising/longRun/fov2/2020-03-09_09.06.55/stack_28_channel_4/SwipeThrough_-450_-210_nimages_49
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200309_Gcamp/fish4_promising/singlePlanes/fov1/2020-03-09_08.41.22/stack_2_channel_4/SinglePlane_-330
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200309_Gcamp/fish4_promising/singlePlanes/fov1/2020-03-09_08.41.22/stack_3_channel_4/SinglePlane_-340
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200309_Gcamp/fish4_promising/singlePlanes/fov1/2020-03-09_08.41.22/stack_4_channel_4/SinglePlane_-350
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200309_Gcamp/fish4_promising/singlePlanes/fov1/2020-03-09_08.41.22/stack_5_channel_4/SinglePlane_-360
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200309_Gcamp/fish4_promising/singlePlanes/fov1/2020-03-09_08.41.22/stack_7_channel_4/SinglePlane_-320
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200309_Gcamp/fish4_promising/singlePlanes/fov1/2020-03-09_08.41.22/stack_8_channel_4/SinglePlane_-310
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200309_Gcamp/fish4_promising/singlePlanes/fov1/2020-03-09_08.41.22/stack_9_channel_4/SinglePlane_-300
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200309_Gcamp/fish4_promising/singlePlanes/fov2/2020-03-09_08.56.54/stack_2_channel_4/SinglePlane_-330
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200309_Gcamp/fish4_promising/singlePlanes/fov2/2020-03-09_09.31.52/stack_10_channel_4/SinglePlane_-290
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200309_Gcamp/fish4_promising/singlePlanes/fov2/2020-03-09_09.31.52/stack_12_channel_4/SinglePlane_-390
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200309_Gcamp/fish4_promising/singlePlanes/fov2/2020-03-09_09.31.52/stack_13_channel_4/SinglePlane_-380
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200309_Gcamp/fish4_promising/singlePlanes/fov2/2020-03-09_09.31.52/stack_14_channel_4/SinglePlane_-280
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200309_Gcamp/fish4_promising/singlePlanes/fov2/2020-03-09_09.31.52/stack_15_channel_4/SinglePlane_-270
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200309_Gcamp/fish4_promising/singlePlanes/fov2/2020-03-09_09.31.52/stack_17_channel_4/SinglePlane_-385
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200309_Gcamp/fish4_promising/singlePlanes/fov2/2020-03-09_09.31.52/stack_18_channel_4/SinglePlane_-375
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200309_Gcamp/fish4_promising/singlePlanes/fov2/2020-03-09_09.31.52/stack_19_channel_4/SinglePlane_-365
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200309_Gcamp/fish4_promising/singlePlanes/fov2/2020-03-09_09.31.52/stack_20_channel_4/SinglePlane_-355
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200309_Gcamp/fish4_promising/singlePlanes/fov2/2020-03-09_09.31.52/stack_21_channel_4/SinglePlane_-345
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200309_Gcamp/fish4_promising/singlePlanes/fov2/2020-03-09_09.31.52/stack_22_channel_4/SinglePlane_-335
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200309_Gcamp/fish4_promising/singlePlanes/fov2/2020-03-09_09.31.52/stack_23_channel_4/SinglePlane_-325
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200309_Gcamp/fish4_promising/singlePlanes/fov2/2020-03-09_09.31.52/stack_24_channel_4/SinglePlane_-315
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200309_Gcamp/fish4_promising/singlePlanes/fov2/2020-03-09_09.31.52/stack_25_channel_4/SinglePlane_-295
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200309_Gcamp/fish4_promising/singlePlanes/fov2/2020-03-09_09.31.52/stack_26_channel_4/SinglePlane_-285
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200309_Gcamp/fish4_promising/singlePlanes/fov2/2020-03-09_09.31.52/stack_27_channel_4/SinglePlane_-275
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200309_Gcamp/fish4_promising/singlePlanes/fov2/2020-03-09_09.31.52/stack_29_channel_4/SinglePlane_-305
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200309_Gcamp/fish4_promising/singlePlanes/fov2/2020-03-09_09.31.52/stack_2_channel_4/SinglePlane_-330
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200309_Gcamp/fish4_promising/singlePlanes/fov2/2020-03-09_09.31.52/stack_3_channel_4/SinglePlane_-340
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200309_Gcamp/fish4_promising/singlePlanes/fov2/2020-03-09_09.31.52/stack_4_channel_4/SinglePlane_-350
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200309_Gcamp/fish4_promising/singlePlanes/fov2/2020-03-09_09.31.52/stack_5_channel_4/SinglePlane_-360
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200309_Gcamp/fish4_promising/singlePlanes/fov2/2020-03-09_09.31.52/stack_6_channel_4/SinglePlane_-370
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200309_Gcamp/fish4_promising/singlePlanes/fov2/2020-03-09_09.31.52/stack_7_channel_4/SinglePlane_-320
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200309_Gcamp/fish4_promising/singlePlanes/fov2/2020-03-09_09.31.52/stack_8_channel_4/SinglePlane_-310
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200309_Gcamp/fish4_promising/singlePlanes/fov2/2020-03-09_09.31.52/stack_9_channel_4/SinglePlane_-300
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200309_Gcamp/fish4_promising/superFast_50Hz/2020-03-09_07.53.40/stack_16_channel_5/SinglePlane_-330
=========================================
fish_11_1,  have to double check, some move a lot....
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish1/5Hz/longRun/2020-03-11_02.34.23/stack_32_channel_4/SwipeThrough_-450_-210_nimages_121
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish1/5Hz/longRun/2020-03-11_02.34.23/stack_33_channel_4/SwipeThrough_-390_-270_nimages_121
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish1/5Hz/longRun/2020-03-11_02.34.23/stack_34_channel_4/SinglePlane_-330
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish1/5Hz/longRun/2020-03-11_02.43.33/stack_32_channel_3/SwipeThrough_-450_-210_nimages_121
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish1/5Hz/longRun/2020-03-11_02.43.33/stack_33_channel_3/SwipeThrough_-390_-270_nimages_121
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish1/5Hz/longRun/2020-03-11_02.48.29/stack_32_channel_8/SwipeThrough_-450_-210_nimages_121
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish1/5Hz/longRun/2020-03-11_02.48.29/stack_33_channel_8/SwipeThrough_-390_-270_nimages_121
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish1/5Hz/longRun/2020-03-11_02.49.27/stack_32_channel_8/SwipeThrough_-450_-210_nimages_121
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish1/5Hz/longRun/2020-03-11_02.49.27/stack_33_channel_8/SwipeThrough_-390_-270_nimages_121
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish1/5Hz/longRun/2020-03-11_02.54.20/stack_32_channel_8/SwipeThrough_-450_-210_nimages_121
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish1/5Hz/longRun/2020-03-11_02.54.20/stack_33_channel_8/SwipeThrough_-390_-270_nimages_121
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish1/5Hz/longRun/2020-03-11_03.22.33/stack_32_channel_4/SwipeThrough_-450_-210_nimages_121
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish1/5Hz/longRun/2020-03-11_03.22.33/stack_33_channel_4/SwipeThrough_-390_-270_nimages_121
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish1/5Hz/longRun/2020-03-11_03.22.33/stack_34_channel_4/SinglePlane_-330
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish1/5Hz/longRun/2020-03-11_06.16.29/stack_32_channel_4/SwipeThrough_-450_-210_nimages_121
# LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish1/5Hz/longRun/leakingBrain/2020-03-11_02.00.31/stack_2_channel_4/SinglePlane_-330
# LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish1/5Hz/longRun/leakingBrain/2020-03-11_02.00.31/stack_32_channel_4/SwipeThrough_-450_-210_nimages_121
# LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish1/5Hz/longRun/leakingBrain/2020-03-11_02.00.31/stack_33_channel_4/SwipeThrough_-390_-270_nimages_121
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish1/5Hz/longRun/niceOne2/2020-03-11_04.03.27/stack_33_channel_4/SwipeThrough_-390_-270_nimages_121
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish1/5Hz/longRun/niceOne2/2020-03-11_04.17.17/stack_33_channel_4/SwipeThrough_-390_-270_nimages_121
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish1/5Hz/longRun/niceOne3/2020-03-11_04.25.22/stack_34_channel_4/SinglePlane_-330
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish1/5Hz/longRun/niceOne3/2020-03-11_04.27.34/stack_33_channel_4/SwipeThrough_-390_-270_nimages_121
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish1/5Hz/longRun/niceOne3/2020-03-11_04.28.39/stack_33_channel_4/SwipeThrough_-390_-270_nimages_121
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish1/5Hz/longRun/niceOne3/2020-03-11_04.35.02/stack_33_channel_4/SwipeThrough_-390_-270_nimages_121
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish1/5Hz/longRun/niceOne4/2020-03-11_04.47.24/stack_32_channel_4/SwipeThrough_-450_-210_nimages_121
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish1/5Hz/longRun/niceOne5/2020-03-11_05.53.13/stack_32_channel_4/SwipeThrough_-450_-210_nimages_121
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish1/5Hz/longRun/niceOne/2020-03-11_03.30.35/stack_32_channel_4/SwipeThrough_-450_-210_nimages_121
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish1/5Hz/longRun/niceOne/2020-03-11_03.30.35/stack_33_channel_4/SwipeThrough_-390_-270_nimages_121
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish1/5Hz/longRun/niceOne/2020-03-11_03.43.42/stack_34_channel_4/SinglePlane_-330
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish1/5Hz/longRun/niceOne/2020-03-11_03.48.47/stack_34_channel_4/SinglePlane_-330
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish1/5Hz/longRun/niceOne/2020-03-11_04.00.37/stack_34_channel_4/SinglePlane_-330
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish1/5Hz/longRun/slideThrough/2020-03-11_05.26.26/stack_35_channel_4/SwipeThrough_-450_-210_nimages_121
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish1/5Hz/longRun/slideThrough/2020-03-11_05.36.29/stack_35_channel_4/SwipeThrough_-450_-210_nimages_121
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish1/5Hz/singlePlanes/2020-03-11_06.07.48/stack_2_channel_4/SinglePlane_-330
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish1/5Hz/singlePlanes/2020-03-11_06.07.48/stack_3_channel_4/SinglePlane_-340
=========================================
fish_11_2
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish2/10Hz/2020-03-11_07.34.47/stack_33_channel_3/SwipeThrough_-390_-270_nimages_121
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish2/10Hz/241Planes/2020-03-11_08.52.33/stack_1_channel_3/SwipeThrough_-450_-210_nimages_241
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish2/10Hz/241Planes/2020-03-11_09.08.00/stack_1_channel_3/SwipeThrough_-450_-210_nimages_241
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish2/10Hz/singlePlane/2020-03-11_07.30.39/stack_7_channel_3/SinglePlane_-320
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish2/10Hz/singlePlane/2020-03-11_07.30.39/stack_8_channel_3/SinglePlane_-310
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish2/10Hz/singlePlane/2020-03-11_10.13.20/stack_10_channel_3/SinglePlane_-290
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish2/10Hz/singlePlane/2020-03-11_10.13.20/stack_24_channel_3/SinglePlane_-315
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish2/10Hz/singlePlane/2020-03-11_10.25.41/stack_25_channel_3/SinglePlane_-295
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish2/10Hz/singlePlane/2020-03-11_10.25.41/stack_29_channel_3/SinglePlane_-305
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish2/10Hz/singlePlane/2020-03-11_10.25.41/stack_3_channel_3/SinglePlane_-340
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish2/10Hz/singlePlane/superNice/2020-03-11_10.17.34/stack_14_channel_3/SinglePlane_-280
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish2/10Hz/singlePlane/superNice/2020-03-11_10.17.34/stack_16_channel_3/SinglePlane_-330
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish2/10Hz/singlePlane/superNice/2020-03-11_10.21.14/stack_25_channel_3/SinglePlane_-295
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish2/10Hz/singlePlane/superNice/2020-03-11_10.21.14/stack_29_channel_3/SinglePlane_-305
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish2/10Hz/slideThrough/2020-03-11_08.30.21/stack_36_channel_3/SwipeThrough_-390_-270_nimages_121
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish2/10Hz/slideThrough/2020-03-11_08.34.19/stack_36_channel_3/SwipeThrough_-390_-270_nimages_121
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish2/10Hz/slideThrough/2020-03-11_08.36.58/stack_36_channel_3/SwipeThrough_-390_-270_nimages_121
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish2/10Hz/slideThrough/2020-03-11_08.39.31/stack_36_channel_3/SwipeThrough_-390_-270_nimages_121
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish2/10Hz/slideThrough/2020-03-11_08.42.07/stack_36_channel_3/SwipeThrough_-390_-270_nimages_121
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish2/10Hz/slideThrough/2020-03-11_08.44.50/stack_36_channel_3/SwipeThrough_-390_-270_nimages_121
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish2/10Hz/slideThrough/2020-03-11_08.47.29/stack_36_channel_3/SwipeThrough_-390_-270_nimages_121
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish2/16ms/2020-03-11_10.33.17/stack_37_channel_9/SinglePlane_-340
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish2/16ms/2020-03-11_10.34.15/stack_37_channel_9/SinglePlane_-340
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish2/16ms/2020-03-11_10.35.41/stack_37_channel_9/SinglePlane_-340
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish2/20Hz/2020-03-11_08.19.37/stack_33_channel_8/SwipeThrough_-390_-270_nimages_121
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish2/20Hz/2020-03-11_08.20.29/stack_33_channel_8/SwipeThrough_-390_-270_nimages_121
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish2/20Hz/2020-03-11_08.22.19/stack_33_channel_8/SwipeThrough_-390_-270_nimages_121
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish2/50Hz/2020-03-11_08.12.13/stack_8_channel_5/SinglePlane_-310
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish2/5Hz/2020-03-11_06.53.14/stack_34_channel_4/SinglePlane_-330
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish2/5Hz/2020-03-11_06.55.38/stack_32_channel_4/SwipeThrough_-450_-210_nimages_121
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish2/5Hz/2020-03-11_06.57.26/stack_33_channel_4/SwipeThrough_-390_-270_nimages_121
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish2/Heinbrain/2020-03-11_10.06.25/stack_2_channel_3/SinglePlane_-330  # todo add to training
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish2/Heinbrain/2020-03-11_10.06.25/stack_33_channel_3/SwipeThrough_-390_-270_nimages_121
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish2/beads_after_fish/2020-03-11_10.45.34/stack_0_channel_0/SinglePlane_-450
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish2/beads_after_fish/2020-03-11_10.45.34/stack_1_channel_1/SwipeThrough_-450_-210_nimages_241
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish2/beads_after_fish/2020-03-11_10.45.34/stack_32_channel_1/SwipeThrough_-450_-210_nimages_121
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish2/beads_after_fish/2020-03-11_10.45.34/stack_33_channel_1/SwipeThrough_-390_-270_nimages_121
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish2/opticTectum/10Hz/2020-03-11_09.35.16/stack_33_channel_3/SwipeThrough_-390_-270_nimages_121
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish2/opticTectum/10Hz/2020-03-11_09.36.25/stack_33_channel_3/SwipeThrough_-390_-270_nimages_121
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish2/opticTectum/10Hz/2020-03-11_09.47.38/stack_32_channel_3/SwipeThrough_-450_-210_nimages_121
LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish2/opticTectum/10Hz/2020-03-11_09.54.35/stack_32_channel_3/SwipeThrough_-450_-210_nimages_121
"""
