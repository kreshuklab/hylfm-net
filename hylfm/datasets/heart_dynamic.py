import argparse
from pathlib import Path

import imageio
import yaml
import warnings

from hylfm.datasets.base import TensorInfo, get_dataset_from_info, N5CachedDatasetFromInfoSubset
from hylfm.datasets.heart_utils import get_transformations, idx2z_slice_241


def get_tensor_info(tag: str, name: str, meta: dict):
    meta = dict(meta)
    assert "z_out" in meta
    assert "nnum" in meta
    assert "interpolation_order" in meta
    assert "scale" in meta
    assert "z_ls_rescaled" in meta
    assert "pred_z_min" in meta
    assert "pred_z_max" in meta

    root = "GKRESHUK"
    insert_singleton_axes_at = [0, 0]
    z_slice = None
    samples_per_dataset = 1
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

    if tag in ["2019-12-02_04.12.36_10msExp", "2019-12-02_03.44.01_5msExp"]:
        transformations = []
        meta["quality"] = 4
        stack, channel = {"2019-12-02_04.12.36_10msExp": (1, 3), "2019-12-02_03.44.01_5msExp": (1, 2)}[tag]
        location = f"LF_partially_restored/LenseLeNet_Microscope/20191202_staticHeart_dynamicHeart/data/{tag}/stack_{stack}_channel_{channel}/"
        if name == "lf":
            location += "TP_*/RC_rectified/Cam_Right_*_rectified.tif"
            transformations += [{"Crop": {"apply_to": name, "crop": [(0, None), (19, None), (0, None)]}}]
        # elif name == "lr":
        #     location = location.replace("LF_partially_restored/", "LF_computed/")
        #     location += "TP_*/RCout/Cam_Right_*.tif"
        #     transformations += [{"Crop": {"apply_to": name, "crop": [(0, None), (0, None), (19, None), (0, None)]}}]
        elif name == "fake_ls":
            location += "Cam_Left_*.h5/Data"
        elif name == "ls_slice":
            location += "Cam_Left_*.h5/Data"
            samples_per_dataset = 241
            z_slice = idx2z_slice_241
        crop_name = "wholeFOV"
        transformations += get_transformations(name, crop_name, meta=meta)
    elif tag in ["2019-12-02_04.12.36_10msExp_short"]:
        transformations = []
        meta["quality"] = 4
        location = f"LF_partially_restored/LenseLeNet_Microscope/20191202_staticHeart_dynamicHeart/data/{tag.replace('_short', '')}/stack_1_channel_3/"
        if name == "lf":
            location += "TP_00000/RC_rectified/Cam_Right_*_rectified.tif"
            transformations += [{"Crop": {"apply_to": name, "crop": [(0, None), (19, None), (0, None)]}}]
        elif name == "lr":
            location = location.replace("LF_partially_restored/", "LF_computed/")
            location += "TP_00000/RCout/Cam_Right_*.tif"
            transformations += [{"Crop": {"apply_to": name, "crop": [(0, None), (0, None), (19, None), (0, None)]}}]
        elif name == "fake_ls":
            location += "Cam_Left_00000.h5/Data"
        elif name == "ls_slice":
            location += "Cam_Left_00000.h5/Data"
            samples_per_dataset = 241
            z_slice = idx2z_slice_241
        crop_name = "wholeFOV"
        transformations += get_transformations(name, crop_name, meta=meta)

    elif tag in ["2019-12-02_23.17.56", "2019-12-02_23.43.24", "2019-12-02_23.50.04", "2019-12-03_00.00.44"]:
        if tag == "2019-12-03_00.00.44":
            raise NotImplementedError("check crop and if 10ms is really there now...")
        # if tag == "2019-12-03_00.00.44":
        #     raise NotImplementedError("10ms is coming, only 5ms available:")
        # location = location.replace("stack_1_channel_3", "stack_1_channel_2")
        meta["quality"] = 3
        location = f"LF_partially_restored/LenseLeNet_Microscope/20191203_dynamic_staticHeart_tuesday/fish1/dynamic/Heart_tightCrop/dynamicImaging1_btw20to160planes/{tag}/stack_1_channel_3/"

        if tag in ["2019-12-02_23.43.24", "2019-12-02_23.50.04", "2019-12-03_00.00.44"]:
            if name == "lf":
                padding = [
                    {
                        "Pad": {
                            "apply_to": name,
                            "pad_width": [[0, 0], [0, 1], [0, 0]],
                            "pad_mode": "lenslets",
                            "nnum": meta["nnum"],
                        }
                    }
                ]
            elif name == "lr":
                raise NotImplementedError("padding for lr")
            else:
                padding = []
        else:
            padding = []

        crop_name = "Heart_tightCrop"
        transformations = padding + get_transformations(name, crop_name, meta=meta)
        if name == "lf":
            location += "TP_*/RC_rectified/Cam_Right_*_rectified.tif"
        # elif name == "lr":
        #     location = location.replace("LF_partially_restored/", "LF_computed/")
        #     location += "TP_*/RCout/Cam_Right_*.tif"
        elif name == "ls_slice":
            location += "Cam_Left_*.h5/Data"
            samples_per_dataset = 241
            z_slice = idx2z_slice_241
        else:
            raise NotImplementedError(name)

    elif tag in ["2019-12-08_23.43.42"]:
        meta["quality"] = 1
        crop_name = "Heart_tightCrop"
        transformations = get_transformations(name, crop_name, meta=meta)
        location = f"LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish1/dynamic/Heart_tightCrop/SlideThroughCompleteStack/{tag}/stack_1_channel_3/"
        if name == "lf":
            location += "TP_*/RC_rectified/Cam_Right_*_rectified.tif"
        # elif name == "lr":
        #     location = location.replace("LF_partially_restored/", "LF_computed/")
        #     location += "TP_*/RCout/Cam_Right_*.tif"
        elif name == "ls_slice":
            location += "Cam_Left_*.h5/Data"
            samples_per_dataset = 241
            z_slice = idx2z_slice_241

    elif tag in ["2019-12-09_04.54.38", "2019-12-09_05.21.16"]:
        crop_name = "Heart_tightCrop"
        transformations = get_transformations(name, crop_name, meta=meta)
        location = f"LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/{tag}/stack_1_channel_3/"
        if name == "lf":
            location += "TP_*/RC_rectified/Cam_Right_*_rectified.tif"
        elif name == "lr":
            if tag == "2019-12-09_04.54.38":
                location = "/scratch/Nils/LF_computed/TP*/RCout/Cam_Right_*.tif"
            else:
                location = location.replace("LF_partially_restored/", "LF_computed/")
                location += "TP_*/RCout/Cam_Right_*.tif"
        elif name == "ls_slice":
            location += "Cam_Left_*.h5/Data"
            samples_per_dataset = 241
            z_slice = idx2z_slice_241

    elif tag in ["2019-12-09_04.54.38_short"]:  #  "2019-12-09_05.21.16_short"
        crop_name = "Heart_tightCrop"
        transformations = get_transformations(name, crop_name, meta=meta)
        location = f"LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/{tag.replace('_short', '')}/stack_1_channel_3/"
        if name == "lf":
            location += "TP_00000/RC_rectified/Cam_Right_*_rectified.tif"
        elif name == "lr":
            location = location.replace("LF_partially_restored/", "LF_computed/")
            location += "TP_00000/RCout/Cam_Right_*.tif"
        elif name == "ls_slice":
            location += "Cam_Left_00000.h5/Data"
            samples_per_dataset = 241
            z_slice = idx2z_slice_241

    elif tag in ["2019-12-09_05.41.14_theGoldenOne"]:
        crop_name = "Heart_tightCrop"
        transformations = get_transformations(name, crop_name, meta=meta)
        location = f"LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/{tag}/stack_1_channel_3/"
        if name == "lf":
            location += "TP_*/RC_rectified/Cam_Right_*_rectified.tif"
        # elif name == "lr":
        #     location = location.replace("LF_partially_restored/", "LF_computed/")
        #     location += "TP_*/RCout/Cam_Right_*.tif"
        elif name == "ls_slice":
            location += "Cam_Left_*.h5/Data"
            samples_per_dataset = 241
            z_slice = idx2z_slice_241

    elif tag in ["plane_100/2019-12-09_05.55.26", "plane_120/2019-12-09_05.53.55"]:
        crop_name = "Heart_tightCrop"
        transformations = get_transformations(name, crop_name, meta=meta)
        location = f"LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.41.14_theGoldenOne/singlePlane_samePos/{tag}/stack_2_channel_3/"
        if name == "lf":
            location += "TP_*/RC_rectified/Cam_Right_*_rectified.tif"
        elif name == "lr":
            location = location.replace("LF_partially_restored/", "LF_computed/")
            location += "TP_*/RCout/Cam_Right_*.tif"
        elif name == "ls_slice":
            location += "Cam_Left_*.h5/Data"
            samples_per_dataset = 200
            plane = int(tag.split("/")[0].split("_")[1])
            z_slice = idx2z_slice_241(plane)

    elif tag in ["plane_100/2019-12-09_05.55.26_short", "plane_120/2019-12-09_05.53.55_short"]:
        crop_name = "Heart_tightCrop"
        transformations = get_transformations(name, crop_name, meta=meta)
        location = f"LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.41.14_theGoldenOne/singlePlane_samePos/{tag.replace('_short', '')}/stack_2_channel_3/"
        if name == "lf":
            location += "TP_00000/RC_rectified/Cam_Right_*_rectified.tif"
        elif name == "lr":
            location = location.replace("LF_partially_restored/", "LF_computed/")
            location += "TP_00000/RCout/Cam_Right_*.tif"
        elif name == "ls_slice":
            location += "*Cam_Left_00000.h5/Data"
            samples_per_dataset = 200
            plane = int(tag.split("/")[0].split("_")[1])
            z_slice = idx2z_slice_241(plane)

    elif tag in [
        "2019-12-09_23.10.02",
        "2019-12-09_23.17.30",
        "2019-12-09_23.19.41",
        "2019-12-10_00.40.09",
        "2019-12-10_00.51.54",
        "2019-12-10_01.03.50",
        "2019-12-10_01.25.44",
    ]:
        crop_name = "Heart_tightCrop"
        transformations = get_transformations(name, crop_name, meta=meta)
        location = f"LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish3/dynamic/Heart_tightCrop/slideThroughStack/{tag}/stack_1_channel_3/"
        if name == "lf":
            location += "TP_*/RC_rectified/Cam_Right_*_rectified.tif"
        # elif name == "lr":
        #     location = location.replace("LF_partially_restored/", "LF_computed/")
        #     location += "TP_*/RCout/Cam_Right_*.tif"
        elif name == "ls_slice":
            location += "Cam_Left_*.h5/Data"
            samples_per_dataset = 241
            z_slice = idx2z_slice_241

    elif tag in ["2019-12-10_02.13.34"]:
        crop_name = "Heart_tightCrop"
        transformations = get_transformations(name, crop_name, meta=meta)
        location = f"LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish3/dynamic/Heart_tightCrop/theGoldenExperiment/SlidingThroughStack_samePos/{tag}/stack_1_channel_3/"
        if name == "lf":
            location += "TP_*/RC_rectified/Cam_Right_*_rectified.tif"
        # elif name == "lr":
        #     location = location.replace("LF_partially_restored/", "LF_computed/")
        #     location += "TP_*/RCout/Cam_Right_*.tif"
        elif name == "ls_slice":
            location += "Cam_Left_*.h5/Data"
            samples_per_dataset = 241
            z_slice = idx2z_slice_241

    elif tag in [
        "plane_090/2019-12-09_04.26.13_5ms",
        "plane_090/2019-12-09_04.26.13_10ms",
        "plane_090/2019-12-09_04.26.55_5ms",
        "plane_090/2019-12-09_04.26.55_10ms",
        "plane_090/2019-12-09_04.28.03_5ms",
        "plane_090/2019-12-09_04.28.03_10ms",
        "plane_120/2019-12-09_04.10.59_5ms",
        "plane_120/2019-12-09_04.10.59_10ms",
        "plane_120/2019-12-09_04.11.56_5ms",
        "plane_120/2019-12-09_04.11.56_10ms",
        "plane_120/2019-12-09_04.13.01_5ms",
        "plane_120/2019-12-09_04.13.01_10ms",
        "plane_150/2019-12-09_04.23.37_5ms",
        "plane_150/2019-12-09_04.23.37_10ms",
        "plane_150/2019-12-09_04.24.22_5ms",
        "plane_150/2019-12-09_04.24.22_10ms",
    ]:
        crop_name = "fast_cropped_8ms"
        transformations = get_transformations(name, crop_name, meta=meta)
        if tag.endswith("_5ms"):
            stack, channel = (2, 11)
        elif tag.endswith("_10ms"):
            stack, channel = (2, 10)
        else:
            raise NotImplementedError(tag)

        tag = tag.replace("_5ms", "").replace("_10ms", "")

        location = f"LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/fast_cropped_8ms/singlePlanes/fullyOpenIris/{tag}/stack_{stack}_channel_{channel}/"
        if name == "lf":
            location += "TP_*/RC_rectified/Cam_Right_*_rectified.tif"
        # elif name == "lr":
        #     location = location.replace("LF_partially_restored/", "LF_computed/")
        #     location += "TP_*/RCout/Cam_Right_*.tif"
        elif name == "ls_slice":
            location += "Cam_Left_*.h5/Data"
            samples_per_dataset = 200
            plane = int(tag.split("/")[0].split("_")[1])
            z_slice = idx2z_slice_241(plane)

    elif tag in [
        "plane_080/2019-12-09_04.02.24_5ms",
        "plane_080/2019-12-09_04.02.24_10ms",
        "plane_080/2019-12-09_04.05.07_irisOpenedComplete_5ms",
        "plane_080/2019-12-09_04.05.07_irisOpenedComplete_10ms",
        "plane_100/2019-12-09_03.44.34_5ms",
        "plane_100/2019-12-09_03.44.34_10ms",
        "plane_100/2019-12-09_03.46.56_5ms",
        "plane_100/2019-12-09_03.46.56_10ms",
        "plane_120/2019-12-09_03.41.23_5ms",
        "plane_120/2019-12-09_03.41.23_10ms",
        "plane_120/2019-12-09_03.42.18_5ms",
        "plane_120/2019-12-09_03.42.18_10ms",
        "plane_140/2019-12-09_03.55.51_5ms",
        "plane_140/2019-12-09_03.55.51_10ms",
        "plane_140/2019-12-09_03.56.44_5ms",
        "plane_140/2019-12-09_03.56.44_10ms",
        "plane_160/2019-12-09_03.58.24_5ms",
        "plane_160/2019-12-09_03.58.24_10ms",
        "plane_160/2019-12-09_03.59.45_5ms",
        "plane_160/2019-12-09_03.59.45_10ms",
    ]:
        crop_name = "fast_cropped_8ms"
        transformations = get_transformations(name, crop_name, meta=meta)
        if tag.endswith("_5ms"):
            stack, channel = (2, 11)
        elif tag.endswith("_10ms"):
            stack, channel = (2, 10)
        else:
            raise NotImplementedError(tag)

        tag = tag.replace("_5ms", "").replace("_10ms", "")

        location = f"LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/fast_cropped_8ms/singlePlanes/{tag}/stack_{stack}_channel_{channel}/"
        if name == "lf":
            location += "TP_*/RC_rectified/Cam_Right_*_rectified.tif"
        elif name == "lr":
            location = location.replace("LF_partially_restored/", "LF_computed/")
            location += "TP_*/RCout/Cam_Right_*.tif"
        elif name == "ls_slice":
            location += "Cam_Left_*.h5/Data"
            samples_per_dataset = 200
            plane = int(tag.split("/")[0].split("_")[1])
            z_slice = idx2z_slice_241(plane)

    elif tag in [
        "plane_020/2019-12-03_02.19.20_125Hz",
        "plane_020/2019-12-03_02.19.20_100Hz",
        "plane_040/2019-12-03_02.16.43_125Hz",
        "plane_040/2019-12-03_02.16.43_100Hz",
        "plane_060/2019-12-03_02.14.24_125Hz",
        "plane_060/2019-12-03_02.14.24_100Hz",
        "plane_080/2019-12-03_02.12.08_125Hz",
        "plane_080/2019-12-03_02.12.08_100Hz",
        "plane_080/pos2/2019-12-03_02.47.30_125Hz",
        "plane_080/pos2/2019-12-03_02.47.30_100Hz",
        "plane_100/2019-12-03_02.09.03_125Hz",
        "plane_100/2019-12-03_02.09.03_100Hz",
        "plane_120/2019-12-03_01.57.26_125Hz",
        "plane_120/2019-12-03_01.57.26_100Hz",
        "plane_140/2019-12-03_02.23.56_125Hz",
        "plane_140/2019-12-03_02.23.56_100Hz",
        "plane_160/2019-12-03_02.26.32_125Hz",
        "plane_160/2019-12-03_02.26.32_100Hz",
        "plane_165/2019-12-03_02.33.46_125Hz",
        "plane_165/2019-12-03_02.33.46_100Hz",
        "plane_165_refocused/2019-12-03_02.38.29_125Hz",
        "plane_165_refocused/2019-12-03_02.38.29_100Hz",
    ]:
        crop_name = "fast_cropped_6ms"
        transformations = get_transformations(name, crop_name, meta=meta)
        if tag.endswith("_125Hz"):
            stack, channel = (2, 8)
        elif tag.endswith("_100Hz"):
            stack, channel = (2, 9)
        else:
            raise NotImplementedError(tag)

        if tag == "plane_100/2019-12-03_02.09.03_125Hz":
            if name == "lf":
                transformations.insert(
                    0, {"Crop": {"apply_to": name, "crop": [(0, None), (0, 931), (0, None)]}}
                )
            elif name == "lr":
                transformations.insert(
                    0, {"Crop": {"apply_to": name, "crop": [(0, None), (0, None), (0, 931), (0, None)]}}
                )

        tag = tag.replace("_125Hz", "").replace("_100Hz", "")

        location = f"LF_partially_restored/LenseLeNet_Microscope/20191203_dynamic_staticHeart_tuesday/fish1/dynamic/fast_cropped_6ms_SinglePlaneValidation/{tag}/stack_{stack}_channel_{channel}/"
        if name == "lf":
            location += "TP_*/RC_rectified/Cam_Right_*_rectified.tif"
        elif name == "lr":
            location = location.replace("LF_partially_restored/", "/scratch/")
            location += "TP_*/RCout/Cam_Right_*.tif"
        elif name == "ls_slice":
            location += "Cam_Left_*.h5/Data"
            samples_per_dataset = 500
            plane = int(tag.split("/")[0].split("_")[1])
            z_slice = idx2z_slice_241(plane)
    else:
        raise NotImplementedError(tag)

    if location is None or location.endswith("/"):
        raise NotImplementedError(f"tag: {tag}, name: {name}")

    assert tag.replace("_short", "") in location or tag == "2019-12-09_04.54.38", (tag, name, location)
    tag = tag.replace("/", "_")
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


def debug():
    tight_heart_bdv = [
        [
            0.97945,
            0.0048391,
            -0.096309,
            -88.5296,
            -0.0074754,
            0.98139,
            0.15814,
            -91.235,
            0.016076,
            0.0061465,
            4.0499,
            -102.0931,
        ]
    ]

    def get_vol_trf(name: str):
        return [
            {
                "Resize": {
                    "apply_to": name,
                    "shape": [1.0, 1.0, 0.21052631578947368421052631578947, 0.21052631578947368421052631578947],
                    # "shape": [1.0, 1.0, 0.42105263157894736842105263157895, 0.42105263157894736842105263157895],
                    "order": 2,
                }
            }
        ]

    # ds_ls = get_dataset_from_info(f20191209_081940_ls, transformations=get_vol_trf("ls"), cache=True, indices=[0])
    # print("len ls", len(ds_ls))

    # ds_ls_trf = get_dataset_from_info(
    #     f20191209_081940_ls_trf, transformations=get_vol_trf("ls_trf"), cache=True, indices=[0]
    # )
    # print("len ls_trf", len(ds_ls_trf))
    # ds_ls_trf = get_dataset_from_info(
    #     f20191209_081940_ls_trf, transformations=[], cache=True, indices=[0]
    # )

    # ds_ls_tif = get_dataset_from_info(
    #     f20191209_081940_ls_tif, transformations=get_vol_trf("ls_tif"), cache=True, indices=[0]
    # )
    # print("len ls tif", len(ds_ls_tif))

    # slice_indices = [0, 1, 2, 3, 4, 40, 80, 120, 160, 200, 240]
    # ds_ls_slice = get_dataset_from_info(
    #     f20191209_081940_ls_slice, transformations=get_vol_trf("ls_slice"), cache=True, indices=slice_indices
    # )

    # print("len ls_tif", len(ds))
    # ls_tif = ds[0]["ls_tif"]
    # print("ls_tif", ls_tif.shape)
    #
    # print("diff max", ls.max(), ls_tif.max())
    # print("max diff", (ls - ls_tif).max())
    #
    #
    # plt.imshow(ls[0, 0].max(0))
    # plt.title("ls0")
    # plt.show()
    # plt.imshow(ls[0, 0].max(1))
    # plt.title("ls1")
    # plt.show()
    #
    # plt.imshow(ls_tif[0, 0].max(0))
    # plt.title("ls_tif0")
    # plt.show()
    # plt.imshow(ls_tif[0, 0].max(1))
    # plt.title("ls_tif1")
    # plt.show()

    # ds_lr = get_dataset_from_info(f20191209_081940_lr, transformations=get_vol_trf("lr"), cache=False, indices=[0])
    # print("len ds_lr", len(ds_lr))
    # ds_lr_repeat = get_dataset_from_info(
    #     f20191209_081940_lr_repeat, transformations=get_vol_trf("lr"), cache=True, indices=slice_indices
    # )
    # print("len ds_lr_repeat", len(ds_lr_repeat))

    # ds_zip_slice = ZipDataset(
    #     datasets={"lr": ds_lr_repeat, "ls_slice": ds_ls_slice},
    #     transformation=AffineTransformation(
    #         apply_to={"lr": "lr_trf"},
    #         target_to_compare_to="ls_slice",
    #         order=2,
    #         ref_input_shape=[838, 1273, 1463],
    #         bdv_affine_transformations=tight_heart_bdv,
    #         ref_output_shape=[241, 1451, 1651],
    #         ref_crop_in=[[0, None], [0, None], [0, None]],
    #         ref_crop_out=[[0, None], [0, None], [0, None]],
    #         inverted=False,
    #         padding_mode="border",
    #     ),
    # )

    # ds_zip = ZipDataset(
    #     datasets={"ls": ds_ls, "lr": ds_lr, "ls_tif": ds_ls_tif},
    #     # transformation=AffineTransformation(
    #     #     apply_to={"lr": "lr_trf"},
    #     #     target_to_compare_to="ls",
    #     #     order=2,
    #     #     ref_input_shape=[838, 1273, 1463],
    #     #     bdv_affine_transformations=tight_heart_bdv,
    #     #     ref_output_shape=[241, 1451, 1651],
    #     #     ref_crop_in=[[0, None], [0, None], [0, None]],
    #     #     ref_crop_out=[[0, None], [0, None], [0, None]],
    #     #     inverted=False,
    #     #     padding_mode="border",
    #     # ),
    # )
    # sample = ds_zip[0]

    # ds_zip = ZipDataset(datasets={"ls_trf": ds_ls_trf, "lr": ds_lr})
    # sample = ds_zip[0]

    # def save_vol(name):
    #     vol = sample[name]
    #     imageio.volwrite(f"/g/kreshuk/LF_computed/lnet/debug_affine_fish_trf/{name}.tif", numpy.squeeze(vol))
    #
    # def plot_vol(name):
    #     vol = sample[name]
    #     fig, ax = plt.subplots(2)
    #     for i in range(2):
    #         ax[i].imshow(vol[0, 0].max(i))
    #         ax[i].set_title(f"{name}_super_big{i}")
    #
    #     plt.show()
    #
    # for name in ["ls_trf"]:
    #     save_vol(name)
    #     plot_vol(name)

    # for idx in range(11):
    #     sample = ds_zip_slice[idx]
    #     fig, ax = plt.subplots(2)
    #     ax[0].imshow(sample["ls_slice"][0, 0, 0])
    #     ax[0].set_title(f"ls_slice idx: {idx} z: {sample['meta'][0]['ls_slice']['z_slice']}")
    #     ax[1].imshow(sample["lr_trf"][0, 0, 0])
    #     ax[1].set_title(f"lr_trf idx: {idx}")
    #     plt.show()


# def get_z_hist(ds: N5CachedDatasetFromInfoSubset):
#     z_slices =
#
#     numpy.histogram(a


def check_filter(tag: str, meta: dict):
    lf_crops = {"Heart_tightCrop": [[0, None], [0, None], [0, None]], "wholeFOV": [[0, None], [0, None], [0, None]]}

    # filters = [
    #     ("z_range", {}),
    #     ("signal2noise", {"apply_to": "ls_slice", "signal_percentile": 99.9, "noise_percentile": 5.0, "ratio": 1.5}),
    # ]
    filters = [
        ("z_range", {}),
        ("signal2noise", {"apply_to": "ls_slice", "signal_percentile": 99.99, "noise_percentile": 5.0, "ratio": 1.2}),
    ]

    ds_unfiltered = get_dataset_from_info(get_tensor_info(tag, "ls_slice", meta=meta), cache=True)
    print(" unfiltered", len(ds_unfiltered))
    ds = get_dataset_from_info(get_tensor_info(tag, "ls_slice", meta=meta), cache=True, filters=filters)

    print("ds filtered", len(ds))

    filters = [
        ("z_range", {}),
        ("signal2noise", {"apply_to": "ls_slice", "signal_percentile": 99.9, "noise_percentile": 5.0, "ratio": 2.0}),
    ]
    ds = get_dataset_from_info(get_tensor_info(tag, "ls_slice", meta=meta), cache=True, filters=filters)
    print("ds filtered", len(ds))


def check_data(tag: str, meta: dict):
    ls_slice = get_dataset_from_info(get_tensor_info(tag, "ls_slice", meta=meta), cache=True)
    if meta["scale"] == 4:
        lf = get_dataset_from_info(get_tensor_info(tag, "lf", meta=meta), cache=True)
        assert len(lf) == len(ls_slice), (tag, len(lf), len(ls_slice))

    assert len(ls_slice) > 0, tag
    print(tag, len(ls_slice))
    # lf = lf[0]["lf"]
    # ls = ls[0]["ls_slice"]
    # print("\tlf", lf.shape)
    # print("\tls", ls.shape)
    # imageio.imwrite(f"/g/kreshuk/LF_computed/lnet/padded_lf_{tag}_pad_at_1.tif", lf[0, 0])
    # path = Path(f"/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191202_staticHeart_dynamicHeart/data/{tag}/stack_1_channel_3/TP_00000/RC_rectified_cropped0/Cam_Right_001_rectified.tif")
    # path.parent.mkdir(parents=True, exist_ok=True)
    # imageio.imwrite(path, lf[0, 0])


def search_data():
    path = Path(
        "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191203_dynamic_staticHeart_tuesday/fish1/dynamic/Heart_tightCrop/dynamicImaging1_btw20to160planes/2019-12-03_00.00.44/stack_1_channel_2"
    )
    for dir in path.glob("*/RC_rectified/"):
        print(dir.parent.name, len(list(dir.glob("*.tif"))))


def get_tags():
    with (Path(__file__).parent / "tags" / Path(__file__).with_suffix(".yml").name).open() as f:
        return [tag.strip() for tag in yaml.safe_load(f)]


def quick_check_all(meta: dict):
    # tags = get_tags()
    # tags = ["2019-12-02_23.17.56", "2019-12-02_23.43.24", "2019-12-02_23.50.04", "2019-12-02_04.12.36_10msExp"]
    # tags = ["2019-12-09_04.54.38", "2019-12-09_05.21.16", "2019-12-09_05.41.14_theGoldenOne", "plane_100/2019-12-09_05.55.26"]
    # tags = ["2019-12-02_04.12.36_10msExp"]
    tags = [
        "plane_080/2019-12-09_04.02.24_5ms",
        "plane_080/2019-12-09_04.05.07_irisOpenedComplete_10ms",
        "plane_100/2019-12-09_03.44.34_5ms",
        "plane_100/2019-12-09_03.46.56_10ms",
        "plane_120/2019-12-09_03.41.23_5ms",
        "plane_120/2019-12-09_03.42.18_10ms",
        "plane_140/2019-12-09_03.55.51_5ms",
        "plane_140/2019-12-09_03.56.44_10ms",
        "plane_160/2019-12-09_03.58.24_5ms",
        "plane_160/2019-12-09_03.58.24_10ms",
        "plane_160/2019-12-09_03.59.45_5ms",
        "plane_160/2019-12-09_03.59.45_10ms"
    ]
    for tag in tags:
        try:
            lf = get_dataset_from_info(get_tensor_info(tag, "lf", meta=meta), cache=True)
        except Exception as e:
            print(tag, e)
            lf = []
        try:
            ls_slice = get_dataset_from_info(get_tensor_info(tag, "ls_slice", meta=meta), cache=True)
        except Exception as e:
            print(tag, e)
            ls_slice = []

        try:
            lr_slice = get_dataset_from_info(get_tensor_info(tag, "lr_slice", meta=meta), cache=False)
        except Exception as e:
            print(tag, e)
            lr_slice = []

        if len(lf) != len(ls_slice) or len(lf) != len(lr_slice) or len(ls_slice) == 0:
            print(tag, len(lf), len(ls_slice), len(lr_slice))
        else:
            print(tag, len(lf))

        # check_filter(tag, meta=meta)


if __name__ == "__main__":
    quick_check_all(
        meta={
            "z_out": 49,
            "nnum": 19,
            "interpolation_order": 2,
            "scale": 4,
            "z_ls_rescaled": 241,
            "pred_z_min": 0,
            "pred_z_max": 838,
        }
    )
