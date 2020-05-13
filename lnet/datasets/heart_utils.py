from lnet.transformations.affine_utils import (
    Heart_tightCrop,
    get_bdv_affine_transformations_by_name,
    get_lf_shape,
    get_ls_shape,
    get_raw_ls_crop,
    get_ref_ls_shape,
    staticHeartFOV,
    wholeFOV,
)


def get_transformations(name: str, crop_name: str, meta: dict):
    assert crop_name in [Heart_tightCrop, staticHeartFOV, wholeFOV]
    if name == "lf":
        return [{"Assert": {"apply_to": name, "expected_tensor_shape": [1, 1] + get_lf_shape(crop_name)}}]
    elif name in ["ls", "ls_trf"]:
        trf = [
            {"Assert": {"apply_to": name, "expected_tensor_shape": [1, 1, 241, 2048, 2060]}},  # raw ls shape
            {"FlipAxis": {"apply_to": name, "axis": 2}},
            {"FlipAxis": {"apply_to": name, "axis": 1}},
            {"Crop": {"apply_to": name, "crop": get_raw_ls_crop(crop_name)}},
            {"Assert": {"apply_to": name, "expected_tensor_shape": [1, 1] + get_ls_shape(crop_name)}},
        ]
        if name == "ls_trf":
            trf += [
                {"Cast": {"apply_to": name, "dtype": "float32", "device": "cpu"}},
                {
                    "AffineTransformation": {
                        "apply_to": name,
                        "target_to_compare_to": [meta["z_out"]]
                        + [xy / meta["nnum"] * meta["scale"] for xy in get_lf_shape(crop_name)],
                        "order": meta["interpolation_order"],
                        "ref_input_shape": [838] + get_lf_shape(crop_name),
                        "bdv_affine_transformations": get_bdv_affine_transformations_by_name(crop_name),
                        "ref_output_shape": get_ref_ls_shape(crop_name),
                        "ref_crop_in": [[0, None], [0, None], [0, None]],
                        "ref_crop_out": get_raw_ls_crop(crop_name, wrt_ref=True),
                        "inverted": True,
                        "padding_mode": "border",
                        "align_corners": meta.get("align_corners", False),
                    }
                },
                {"Cast": {"apply_to": name, "dtype": "float32", "device": "numpy"}},
            ]
        else:
            trf += [
                {
                    "Resize": {
                        "apply_to": name,
                        "shape": [
                            1.0,
                            meta["z_ls_rescaled"],
                            meta["scale"] / meta["nnum"],
                            meta["scale"] / meta["nnum"],
                        ],
                        "order": meta["interpolation_order"],
                    }
                },
                {"Cast": {"apply_to": name, "dtype": "float32", "device": "numpy"}},
                {
                    "Assert": {
                        "apply_to": name,
                        "expected_tensor_shape": [None, 1, meta["z_ls_rescaled"]]
                        + [s / meta["nnum"] * meta["scale"] for s in get_ls_shape(crop_name)[1:]],
                    }
                },
            ]

        return trf
    elif name in ["ls_slice", "ls_fake_slice"]:
        return [
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
    elif name == "lr":
        return [
            {"Assert": {"apply_to": name, "expected_tensor_shape": [1, 1, meta["z_out"]] + get_lf_shape(crop_name)}},
            {
                "Resize": {
                    "apply_to": name,
                    "shape": [1.0, 1.0, meta["scale"] / meta["nnum"], meta["scale"] / meta["nnum"]],
                    "order": meta["interpolation_order"],
                }
            },
            {"Cast": {"apply_to": name, "dtype": "float32", "device": "numpy"}},
        ]
    elif name == "ls_reg":
        return [
            {"Assert": {"apply_to": name, "expected_tensor_shape": [1, 1, 838] + get_lf_shape(crop_name)}},  # raw tif
            {
                "Resize": {
                    "apply_to": name,
                    "shape": [1.0, meta["z_out"], meta["scale"] / meta["nnum"], meta["scale"] / meta["nnum"]],
                    "order": meta["interpolation_order"],
                }
            },
            {"Cast": {"apply_to": name, "dtype": "float32", "device": "numpy"}},
        ]

    raise NotImplementedError(f"name: {name}, crop name: {crop_name}")


def idx2z_slice_241(idx: int) -> int:
    return 240 - (idx % 241)
