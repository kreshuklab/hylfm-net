from typing import List, Tuple, Optional

Heart_tightCrop = "Heart_tightCrop"
staticHeartFOV = "staticHeartFOV"


def get_lf_shape(crop_name: str) -> List[int]:
    if crop_name == Heart_tightCrop:
        return [1273, 1463]  # crop on raw (1551,1351) in matlab:
    elif crop_name == staticHeartFOV:
        return [1178, 1767]  # crop on raw in matplab: rect_LF = [100, 400, 1850, 1250]; %[xmin, ymin, width, height]
    else:
        raise NotImplementedError(crop_name)


def get_ref_ls_shape(crop_name: str) -> List[int]:
    if crop_name == Heart_tightCrop:
        return [241, 1451, 1651]
    elif crop_name == staticHeartFOV:
        return [241, 1451, 1951]
    else:
        raise NotImplementedError(crop_name)


def get_raw_ls_crop(crop_name: str, *, for_slice: bool = False, wrt_ref: bool = False) -> List[List[Optional[int]]]:
    """
    crop raw ls shape to be divisible by nnum=19 in yx in order to avoid rounding errors when resizing with scale/nnum
    crop 1 in z, to have many divisors for 240 z planes. (241 is prime)
    """
    if wrt_ref:
        if crop_name == Heart_tightCrop:
            # crop in matlab: 200, 250, 1650, 1450 for ref shape
            # crop for ref shape + crop for divisibility
            ret = [[0, 240], [3, 1700 - 249 - 4], [8, 1850 - 199 - 9]]
        elif crop_name == staticHeartFOV:
            # crop in matlab: 50, 300, 1950, 1450 for ref shape
            # crop for ref shape + crop for divisibility
            ret = [[0, 240], [3, 1750 - 299 - 4], [6, 2000 - 49 - 7]]
        else:
            raise NotImplementedError(crop_name)
    else:
        if crop_name == Heart_tightCrop:
            # crop in matlab: 200, 250, 1650, 1450 for ref shape
            # crop for ref shape + crop for divisibility
            ret = [[0, None], [0, 241 - 1], [249 + 3, 1700 - 4], [199 + 8, 1850 - 9]]
        elif crop_name == staticHeartFOV:
            # crop in matlab: 50, 300, 1950, 1450 for ref shape
            # crop for ref shape + crop for divisibility
            ret = [[0, None], [0, 241 - 1], [299 + 3, 1750 - 4], [49 + 6, 2000 - 7]]
        else:
            raise NotImplementedError(crop_name)

    if for_slice:
        ret[1] = (0, None)

    return ret


def get_ls_shape(crop_name: str) -> List[int]:
    crop = get_raw_ls_crop(crop_name)
    s = [c[1] - c[0] for c in crop[1:]]
    assert len(s) == 3
    assert s[0] == 240
    assert s[1] % 19 == 0
    assert s[2] % 19 == 0
    return s


def get_transformations(name: str, crop_name: str, meta: dict):
    assert crop_name in [Heart_tightCrop, staticHeartFOV]
    if name == "lf":
        return [{"Assert": {"apply_to": name, "expected_tensor_shape": (1, 1) + get_lf_shape(crop_name)}}]
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
                {
                    "AffineTransformation": {
                        "apply_to": name,
                        "target_to_compare_to": [meta["z_out"]]
                        + [xy // meta["nnum"] * meta["scale"] for xy in get_lf_shape(crop_name)],
                        "order": meta["interpolation_order"],
                        "ref_input_shape": [838] + get_lf_shape(crop_name),
                        "bdv_affine_transformations": crop_name,
                        "ref_output_shape": get_ref_ls_shape(crop_name),
                        "ref_crop_in": [[0, None], [0, None], [0, None]],
                        "ref_crop_out": get_raw_ls_crop(crop_name),
                        "inverted": True,
                        "padding_mode": "border",
                    }
                }
            ]
        else:
            trf += [
                {
                    "Resize": {
                        "apply_to": name,
                        "shape": [1.0, meta["z_out"], meta["scale"] / meta["nnum"], meta["scale"] / meta["nnum"]],
                        "order": meta["interpolation_order"],
                    }
                },
                {
                    "Assert": {
                        "apply_to": name,
                        "expected_tensor_shape": [None, 1, meta["z_out"]]
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
            {"Assert": {"apply_to": name, "expected_tensor_shape": [1, 1, 1] + get_ls_shape(crop_name)[1:]}},
        ]
    elif name == "lr":
        return [
            {"Assert": {"apply_to": name, "expected_tensor_shape": [1, 1, 49] + get_lf_shape(crop_name)}},
            {
                "Resize": {
                    "apply_to": name,
                    "shape": [1.0, meta["z_out"], meta["scale"] / meta["nnum"], meta["scale"] / meta["nnum"]],
                    "order": meta["interpolation_order"],
                }
            },
        ]

    raise NotImplementedError(f"name: {name}, crop name: {crop_name}")


def idx2z_slice_241(idx: int) -> int:
    return 240 - (idx % 241)
