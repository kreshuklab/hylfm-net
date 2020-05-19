from lnet.transformations.affine_utils import (
    Heart_tightCrop,
    get_bdv_affine_transformations_by_name,
    get_lf_shape,
    get_ls_shape,
    staticHeartFOV,
    wholeFOV,
    get_precropped_ls_roi_in_raw_ls,
    get_precropped_ls_shape,
    get_raw_lf_shape,
    get_ls_ref_shape,
)


def get_transformations(name: str, crop_name: str, meta: dict):
    assert crop_name in [Heart_tightCrop, staticHeartFOV, wholeFOV]
    if name == "lf":
        return [
            {"Assert": {"apply_to": name, "expected_tensor_shape": [1, 1] + get_raw_lf_shape(crop_name, wrt_ref=True)}}
        ]
    elif name in ["ls", "ls_trf"]:
        trf = [
            {"Assert": {"apply_to": name, "expected_tensor_shape": [1, 1, 241, 2048, 2060]}},  # raw ls shape
            {"FlipAxis": {"apply_to": name, "axis": 2}},
            {"FlipAxis": {"apply_to": name, "axis": 1}},
            {
                "Crop": {
                    "apply_to": name,
                    "crop": [[0, None]] + get_precropped_ls_roi_in_raw_ls(crop_name, for_slice=False, wrt_ref=False),
                }
            },
            {
                "Assert": {
                    "apply_to": name,
                    "expected_tensor_shape": [1, 1]
                    + get_precropped_ls_shape(crop_name, for_slice=False, nnum=meta["nnum"], wrt_ref=True),
                }
            },
        ]
        if name == "ls_trf":
            trf += [
                {"Cast": {"apply_to": name, "dtype": "float32", "device": "cpu"}},
                {
                    "AffineTransformation": {
                        "apply_to": name,
                        "target_to_compare_to": [meta["z_out"]]
                        + get_raw_lf_shape(crop_name, nnum=meta["nnum"], scale=meta["scale"], wrt_ref=False),
                        "order": meta["interpolation_order"],
                        "ref_input_shape": [838] + get_raw_lf_shape(crop_name, wrt_ref=True),
                        "bdv_affine_transformations": get_bdv_affine_transformations_by_name(crop_name),
                        "ref_output_shape": get_ls_ref_shape(crop_name),
                        "ref_crop_in": [[0, None], [0, None], [0, None]],
                        "ref_crop_out": [[0, None], [0, None], [0, None]],
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
                            meta.get("ls_scale", meta["scale"]) / meta["nnum"],
                            meta.get("ls_scale", meta["scale"]) / meta["nnum"],
                        ],
                        "order": meta["interpolation_order"],
                    }
                },
                {"Cast": {"apply_to": name, "dtype": "float32", "device": "numpy"}},
                {
                    "Assert": {
                        "apply_to": name,
                        "expected_tensor_shape": [None, 1, meta["z_ls_rescaled"]]
                        + get_precropped_ls_shape(
                            crop_name,
                            nnum=meta["nnum"],
                            ls_scale=meta.get("ls_scale", meta["scale"]),
                            for_slice=False,
                            wrt_ref=False,
                        )[1:],
                    }
                },
            ]

        return trf
    elif name in ["ls_slice", "ls_fake_slice"]:
        return [
            {"Assert": {"apply_to": name, "expected_tensor_shape": [1, 1, 1, 2048, 2060]}},  # raw ls shape
            {"FlipAxis": {"apply_to": name, "axis": 2}},
            {
                "Crop": {
                    "apply_to": name,
                    "crop": [[0, None]] + get_precropped_ls_roi_in_raw_ls(crop_name, for_slice=True, wrt_ref=False),
                }
            },
            {
                "Assert": {
                    "apply_to": name,
                    "expected_tensor_shape": [1, 1]
                    + get_precropped_ls_shape(crop_name, for_slice=True, nnum=meta["nnum"], wrt_ref=True),
                }
            },
            {
                "Resize": {
                    "apply_to": name,
                    "shape": [
                        1.0,
                        1.0,
                        meta.get("ls_scale", meta["scale"]) / meta["nnum"],
                        meta.get("ls_scale", meta["scale"]) / meta["nnum"],
                    ],
                    "order": meta["interpolation_order"],
                }
            },
            {"Cast": {"apply_to": name, "dtype": "float32", "device": "numpy"}},
            {
                "Assert": {
                    "apply_to": name,
                    "expected_tensor_shape": [1, 1]
                    + get_precropped_ls_shape(
                        crop_name,
                        for_slice=True,
                        nnum=meta["nnum"],
                        ls_scale=meta.get("ls_scale", meta["scale"]),
                        wrt_ref=False,
                    ),
                }
            },
        ]
    elif name == "lr":
        return [
            {
                "Assert": {
                    "apply_to": name,
                    "expected_tensor_shape": [1, 1, meta["z_out"]] + get_raw_lf_shape(crop_name, wrt_ref=True),
                }
            },
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
