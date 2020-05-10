from collections import OrderedDict
from pathlib import Path

import imageio
import numpy
from lnet.transformations import Crop

from lnet.datasets import ZipDataset, get_dataset_from_info, get_tensor_info
from lnet.transformations.affine_utils import get_crops, get_lf_shape, get_ls_shape, get_ref_ls_shape
from lnet.transformations.utils import get_composed_transformation_from_config

import matplotlib.pyplot as plt


def save_vol(sample, name, setting_name):
    vol = sample[name]
    Path(f"/g/kreshuk/LF_computed/lnet/verify_affine_trf_on_beads/{setting_name}").mkdir(parents=True, exist_ok=True)
    imageio.volsave(
        f"/g/kreshuk/LF_computed/lnet/verify_affine_trf_on_beads/{setting_name}/{name}.tif",
        numpy.squeeze(vol),
        bigtiff=True,
    )


def plot_vol(sample, name):
    vol = sample[name]
    fig, ax = plt.subplots(nrows=3)
    for i in range(3):
        ax[i].imshow(vol[0, 0].max(i))
        ax[i].set_title(f"{name}{i}")

    plt.show()


def test_ls_vs_ls_tif():
    meta = {
        "z_out": 49,
        "z_ls_rescaled": 241,
        "nnum": 19,
        "interpolation_order": 2,
        "scale": 2,
        "shrink": 6,
        "ls_z_min": 0,
        "ls_z_max": 241,
        "pred_z_min": 0,
        "pred_z_max": 838,
    }

    ls = get_dataset_from_info(get_tensor_info("heart_static.beads_ref_Heart_tightCrop", "ls", meta), cache=False)
    ls_tif = get_dataset_from_info(
        get_tensor_info("heart_static.beads_ref_Heart_tightCrop_tif", "ls", meta), cache=False
    )

    ls = ls[0]["ls"][0, 0]
    ls_tif = ls_tif[0]["ls"][0, 0]

    Path("/g/kreshuk/LF_computed/lnet/debug_affine_trf_on_beads/Heart_tightCrop").mkdir(exist_ok=True)
    imageio.volwrite(
        f"/g/kreshuk/LF_computed/lnet/debug_affine_trf_on_beads/Heart_tightCrop/ls_osize.tif", numpy.squeeze(ls)
    )
    imageio.volwrite(
        f"/g/kreshuk/LF_computed/lnet/debug_affine_trf_on_beads/Heart_tightCrop/ls_osize_tif.tif", numpy.squeeze(ls_tif)
    )


def test_crop(crop_name: str, pred: str, target_to_compare_to: str, sample: dict, meta: dict):
    setting_name = f"f4_{crop_name}"

    # fake network shrinkage:
    sample = Crop(
        apply_to=pred, crop=[(0, None), (0, None), (meta["shrink"], -meta["shrink"]), (meta["shrink"], -meta["shrink"])]
    )(sample)

    lf_crops = {"Heart_tightCrop": [[0, None]] * 3, "wholeFOV": [[0, None]] * 3}

    trf_config = [
        {
            "Assert": {
                "apply_to": pred,
                "expected_tensor_shape": [None, 1, meta["z_out"]]
                + [s / meta["nnum"] * meta["scale"] - 2 * meta["shrink"] for s in get_lf_shape(crop_name)],
            }
        },
        {
            "Assert": {
                "apply_to": target_to_compare_to,
                "expected_tensor_shape": [None, 1, meta["z_ls_rescaled"]]
                + [s / meta["nnum"] * meta["scale"] for s in get_ls_shape(crop_name)[1:]],
            }
        },
        {"CropLSforDynamicTraining": {"apply_to": target_to_compare_to, "lf_crops": lf_crops, "meta": meta}},
        {"SetPixelValue": {"apply_to": pred, "value": 1.0}},
        {"Cast": {"apply_to": pred, "dtype": "float32", "device": "cuda"}},
        {
            "AffineTransformationDynamicTraining": {
                "apply_to": {pred: pred + "_trf"},
                "target_to_compare_to": target_to_compare_to,
                "lf_crops": lf_crops,
                "meta": meta,
                "padding_mode": "zeros",
            }
        },
        # {
        #     "AffineTransformation": {
        #         "apply_to": {pred: pred + "_trf"},
        #         "target_to_compare_to": target_to_compare_to,
        #         "order": meta["interpolation_order"],
        #         "ref_input_shape": [838] + get_lf_shape(crop_name),
        #         "bdv_affine_transformations": crop_name,
        #         "ref_output_shape": get_ref_ls_shape(crop_name),
        #         "ref_crop_in": ref_crop_in,
        #         "ref_crop_out": ref_crop_out,
        #         "inverted": False,
        #         "padding_mode": "border",
        #     }
        # },
        {"Cast": {"apply_to": target_to_compare_to, "dtype": "float32", "device": "numpy"}},
        {"Cast": {"apply_to": pred, "dtype": "float32", "device": "numpy"}},
        {"Cast": {"apply_to": pred + "_trf", "dtype": "float32", "device": "numpy"}},
    ]

    trf = get_composed_transformation_from_config(trf_config)
    sample = trf(sample)

    for name in [pred, target_to_compare_to, pred + "_trf"]:
        print(name, sample[name].shape)
        save_vol(sample, name, setting_name)
        plot_vol(sample, name)


def test_inverse_trf(crop_name: str, meta: dict):
    ls_reg = "ls_reg"
    ls_trf = "ls_trf"
    ds = ZipDataset(
        OrderedDict(
            [
                (
                    ls_reg,
                    get_dataset_from_info(
                        get_tensor_info("heart_static.beads_ref_Heart_tightCrop_tif", ls_reg, meta), cache=True
                    ),
                ),
                (
                    ls_trf,
                    get_dataset_from_info(
                        get_tensor_info("heart_static.beads_ref_Heart_tightCrop", ls_trf, meta), cache=False
                    ),
                ),
            ]
        ),
        join_dataset_masks=False,
    )
    sample = ds[0]

    setting_name = f"inverse_trf_test_2_{crop_name}"

    for name in [ls_reg, ls_trf]:
        print(name, sample[name].shape)
        save_vol(sample, name, setting_name)
        plot_vol(sample, name)


def find_crop(crop_name: str, pred: str, target_to_compare_to: str, sample: dict, meta: dict):
    # fake network shrinkage:
    sample = Crop(
        apply_to=pred, crop=[(0, None), (0, None), (meta["shrink"], -meta["shrink"]), (meta["shrink"], -meta["shrink"])]
    )(sample)

    trf_config = [
        {
            "Assert": {
                "apply_to": pred,
                "expected_tensor_shape": [None, 1, meta["z_out"]]
                + [s / meta["nnum"] * meta["scale"] - 2 * meta["shrink"] for s in get_lf_shape(crop_name)],
            }
        },
        {
            "Assert": {
                "apply_to": target_to_compare_to,
                "expected_tensor_shape": [None, 1, meta["z_ls_rescaled"]]
                + [s / meta["nnum"] * meta["scale"] for s in get_ls_shape(crop_name)[1:]],
            }
        },
        {"SetPixelValue": {"apply_to": pred, "value": 1.0}},
        # {"Cast": {"apply_to": pred, "dtype": "float32", "device": "cuda"}},
        {
            "AffineTransformation": {
                "apply_to": {pred: pred + "_trf"},
                "target_to_compare_to": target_to_compare_to,
                "order": meta["interpolation_order"],
                "ref_input_shape": [838] + get_lf_shape(crop_name),
                "bdv_affine_transformations": crop_name,
                "ref_output_shape": get_ref_ls_shape(crop_name),
                "ref_crop_in": None,
                "ref_crop_out": None,
                "inverted": False,
                "padding_mode": "zeros",
            }
        },
        {"Cast": {"apply_to": target_to_compare_to, "dtype": "float32", "device": "numpy"}},
        {"Cast": {"apply_to": pred, "dtype": "float32", "device": "numpy"}},
        {"Cast": {"apply_to": pred + "_trf", "dtype": "float32", "device": "numpy"}},
    ]

    trf = get_composed_transformation_from_config(trf_config)
    sample = trf(sample)

    for name in [pred, target_to_compare_to, pred + "_trf"]:
        print(name, sample[name].shape)
        save_vol(sample, name, setting_name)
        plot_vol(sample, name)


if __name__ == "__main__":
    crop_name = "Heart_tightCrop"  # "wholeFOV" "Heart_tightCrop"
    fake_data = True
    meta = {
        "z_out": 419,
        "nnum": 19,
        "scale": 8,
        "interpolation_order": 2,
        "z_ls_rescaled": 241,
        "pred_z_min": 0,
        "pred_z_max": 838,
        "crop_names": ["Heart_tightCrop", "wholeFOV"],
        "shrink": 8,
    }

    pred = "ls_reg"
    target_to_compare_to = "ls"
    if crop_name == "Heart_tightCrop":
        info_name = "heart_static.beads_ref_Heart_tightCrop_tif"
    elif crop_name == "staticHeartFOV":
        info_name = "heart_static.2019-12-08_06.57.57"
    elif crop_name == "wholeFOV":
        info_name = "heart_dynamic.2019-12-02_04.12.36_10msExp"
        target_to_compare_to = "fake_ls"
    else:
        raise NotImplementedError

    setting_name = f"find_f4_tiny_{crop_name}"

    if fake_data:
        sample = {}
        sample[pred] = numpy.ones(
            [1, 1, meta["z_out"]] + [s // meta["nnum"] * meta["scale"] for s in get_lf_shape(crop_name)]
        )
        sample[target_to_compare_to] = numpy.ones(
            [1, 1, meta["z_ls_rescaled"]] + [s // meta["nnum"] * meta["scale"] for s in get_ls_shape(crop_name)[1:]]
        )
        sample["meta"] = [{pred: {"crop_name": crop_name}, target_to_compare_to: {"crop_name": crop_name}}]

    else:
        ds = ZipDataset(
            OrderedDict(
                [
                    (pred, get_dataset_from_info(get_tensor_info(info_name, pred, meta), cache=True)),
                    (
                        target_to_compare_to,
                        get_dataset_from_info(get_tensor_info(info_name, target_to_compare_to, meta), cache=True),
                    ),
                ]
            ),
            join_dataset_masks=False,
        )
        sample = ds[0]

    # test_ls_vs_ls_tif()
    # find_crop(crop_name, pred=pred, target_to_compare_to=target_to_compare_to, sample=sample, meta=meta)
    # test_crop(crop_name, pred=pred, target_to_compare_to=target_to_compare_to, sample=sample, meta=meta)
    test_inverse_trf(crop_name, meta=meta)
