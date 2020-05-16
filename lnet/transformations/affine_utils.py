import warnings
from typing import List, Optional, Sequence, Tuple


def get_bdv_affine_transformations_by_name(name: str) -> List[List[float]]:
    if name == "bead_ref0":
        return [
            [
                0.98048,
                0.004709,
                0.098297,
                -111.7542,
                7.6415e-05,
                0.97546,
                0.0030523,
                -20.1143,
                0.014629,
                8.2964e-06,
                -3.9928,
                846.8515,
            ]
        ]

    elif name == "Heart_tightCrop":
        return [
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

    elif name == "staticHeartFOV":
        return [
            [
                1.000045172184472,
                -6.440948265626484e-4,
                -0.0037246544505502403,
                1.6647525184522693,
                -3.741111751453333e-4,
                0.9997241695263583,
                -7.727988497216694e-6,
                0.5482936082360137,
                6.417439009031318e-4,
                7.834754261221826e-5,
                1.0024816523664135,
                -2.0884853522301463,
            ],
            [
                1.0031348487012806,
                -2.4393612341215746e-4,
                -0.022354095904371995,
                5.848116160919745,
                -5.688306131898453e-4,
                1.0035215202352126,
                0.005454826549562322,
                -2.643832484309726,
                0.009525454800378438,
                -0.0040831532456764375,
                1.0083740999442286,
                -4.757593435405894,
            ],
            [
                0.97669,
                0.0076755,
                0.0042258,
                -95.112,
                -0.0061276,
                0.97912,
                0.03892,
                -134.1098,
                0.007308,
                0.0073582,
                1.1682,
                -92.7323,
            ],
            [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 3.4185, 0.0],
        ]
    elif name == "wholeFOV":
        return [
            [
                1.0000330417435905,
                0.0013576823539939468,
                -0.004114761682115554,
                0.5965954463114479,
                -2.608162465935266e-4,
                0.9997237146596126,
                -0.0012405487854156773,
                0.6049313296603196,
                -0.007468326192493499,
                -0.006393449585647912,
                1.0139671972757003,
                4.030886094615941,
            ],
            [
                1.0000195897268394,
                -0.001326713750716928,
                0.003542923090774597,
                -0.45057279277200163,
                1.959813002048347e-4,
                1.000282458610685,
                0.001418849770725325,
                -0.6307709082217439,
                0.007605383948124034,
                0.006427628247973187,
                0.9844683934402705,
                -3.5619099663579843,
            ],
            [
                1.0002542885875119,
                0.001640391796268422,
                -0.004916804343382724,
                0.7943935244074773,
                -0.0011763878931262618,
                0.9996893129866766,
                0.0020015643710407914,
                0.23570851330359974,
                -0.008752090572813771,
                -0.0064692182918879015,
                0.9973676871201351,
                8.374516246903323,
            ],
            [
                0.9987648848651456,
                -7.348386539296854e-4,
                -0.010270040762137386,
                15.90851688342657,
                5.988465343535266e-4,
                1.0031490069810332,
                1.7766079327323015e-4,
                -16.00168945475245,
                -0.0010431335860339346,
                0.01921986326427179,
                1.02866818001188,
                -24.853086195115633,
            ],
            [
                0.9998763934501763,
                -0.0010891310450819906,
                0.00253370847820035,
                -0.6115113860549377,
                7.675130470486004e-4,
                0.9983682901916651,
                -7.40670235632776e-4,
                0.8275234429996294,
                -0.00947710327402089,
                0.0038009121698615866,
                0.9993213751411238,
                3.1765203213020143,
            ],
            [
                1.0001499960607572,
                0.0014027361439267177,
                -0.0017760750843483504,
                -0.10921947468546678,
                -1.4673922853670316e-4,
                0.9992407191400258,
                -0.0017297884331933565,
                0.9394132446795214,
                0.024767938375552963,
                -0.00398032235341257,
                0.9783021841535382,
                -7.444609223601558,
            ],
            [
                1.0003559629383398,
                2.9327823350429983e-4,
                6.428006590138689e-4,
                -0.5171440459931798,
                -4.050787570647998e-4,
                0.9996835486510609,
                -2.6101507441633878e-5,
                0.3307699115452033,
                -0.007875800046626481,
                -0.002452377176303182,
                0.9930269401854842,
                7.089568501996945,
            ],
            [
                1.0008586374814403,
                0.0017016621421833808,
                0.0011200077040748062,
                -0.2440484013785469,
                4.817641577129386e-4,
                0.9983909483762317,
                0.002322117596739517,
                -1.191038991556635,
                -0.012612826427677736,
                0.0017988091415114254,
                1.0142720444336737,
                6.527121708955718,
            ],
            [
                0.97958,
                0.0047483,
                -0.01109,
                -151.2572,
                -0.0074967,
                0.98373,
                0.049058,
                -134.7033,
                0.013631,
                -0.0030357,
                1.1662,
                -75.4176,
            ],
            [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 3.4185, 0.0],
        ]
    elif name == "gcamp":
        return [[0.98048,0.004709,0.098297,-111.7542,7.6415e-05,0.97546,0.0030523,-20.1143,0.014629,8.2964e-06,-3.9928,846.8515]]
    else:
        raise NotImplementedError(name)


Heart_tightCrop = "Heart_tightCrop"
staticHeartFOV = "staticHeartFOV"
wholeFOV = "wholeFOV"
gcamp = "gcamp"


def get_lf_shape(crop_name: str) -> List[int]:
    if crop_name == Heart_tightCrop:
        return [1273, 1463]  # crop on raw (1551,1351) in matlab:
    elif crop_name == staticHeartFOV:
        return [1178, 1767]  # crop on raw in matlab: rect_LF = [100, 400, 1850, 1250]; %[xmin, ymin, width, height]
    elif crop_name == wholeFOV:
        return [1064, 1083]  # crop on raw in matlab: rect_LF = [450, 450, 1150, 1150]; %[xmin, ymin, width, height];
    elif crop_name == gcamp:
        return [
            1330,
            1615,
        ]  # crop on raw in matlab: rect_LF = [174, 324, 1700, 1400]; %[250, 300, 1550, 1350]; %[xmin, ymin, width, height];
    else:
        raise NotImplementedError(crop_name)


def get_ref_ls_shape(crop_name: str) -> List[int]:
    if crop_name == Heart_tightCrop:
        return [241, 1451, 1651]
    elif crop_name == staticHeartFOV:
        return [241, 1451, 1951]
    elif crop_name == wholeFOV:
        return [241, 1351, 1351]
    elif crop_name == gcamp:
        return [241, 1774, 1924]
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
            ret = [[0, 241], [3, 1700 - 249 - 4], [8, 1850 - 199 - 9]]
        elif crop_name == staticHeartFOV:
            # crop in matlab: 50, 300, 1950, 1450 for ref shape
            # crop for ref shape + crop for divisibility
            ret = [[0, 241], [3, 1750 - 299 - 4], [6, 2000 - 49 - 7]]
        elif crop_name == wholeFOV:
            # crop in matlab: # rect_LS = [350, 350, 1350, 1350]; %[xmin, ymin, width, height];
            # crop for ref shape + crop for divisibility
            ret = [[0, 241], [1, 1700 - 349 - 1], [1, 1700 - 349 - 1]]
        elif crop_name == gcamp:
            # in matlab: 124, 274, 1800, 1500
            ret = [[0, 241], [0, 1774 - 273], [7, 1924 - 123 - 8]]
        else:
            raise NotImplementedError(crop_name)
    else:
        if crop_name == Heart_tightCrop:
            # crop in matlab: 200, 250, 1650, 1450 for ref shape
            # crop for ref shape + crop for divisibility
            ret = [[0, None], [0, 241], [249 + 3, 1700 - 4], [199 + 8, 1850 - 9]]
        elif crop_name == staticHeartFOV:
            # crop in matlab: 50, 300, 1950, 1450 for ref shape
            # crop for ref shape + crop for divisibility
            ret = [[0, None], [0, 241], [299 + 3, 1750 - 4], [49 + 6, 2000 - 7]]
        elif crop_name == wholeFOV:
            # crop in matlab: # rect_LS = [350, 350, 1350, 1350]; %[xmin, ymin, width, height];
            # crop for ref shape + crop for divisibility
            ret = [[0, None], [0, 241], [349 + 1, 1700 - 1], [349 + 1, 1700 - 1]]
        elif crop_name == gcamp:
            ret = [[0, None], [0, 241], [273, 1774], [123 + 7, 1924 - 8]]
        else:
            raise NotImplementedError(crop_name)

    if for_slice:
        ret[1] = (0, None)

    return ret


def get_ls_shape(crop_name: str, for_slice: bool = False) -> List[int]:
    crop = get_raw_ls_crop(crop_name)
    s = [c[1] - c[0] for c in crop[1:]]
    assert len(s) == 3
    assert s[0] == 241
    assert s[1] % 19 == 0, (s[1], s[1] % 19)
    assert s[2] % 19 == 0, (s[2], s[2] % 19)

    if for_slice:
        s[0] = 1

    return s


def get_lf_crop(crop_name: str, *, shrink: int, nnum: int, scale: int) -> List[List[int]]:
    if crop_name in [Heart_tightCrop, staticHeartFOV, wholeFOV]:
        crop_float = [
            [shrink * nnum / scale, lfs - shrink * nnum / scale] for lfs in get_lf_shape(crop_name)
        ]  # loading precropped `rectified.tif`s
        crop = [[int(cc) for cc in c] for c in crop_float]
        assert crop_float == crop, (crop_float, crop)

    elif crop_name == gcamp:  # cropping additionally on top of the precropped `rectified.tif's
        base_crop = [
            [133, 1121],
            [323, 1596],
        ]  # crop on raw in matlab: rect_LF = [174, 324, 1700, 1400]; %[250, 300, 1550, 1350]; %[xmin, ymin, width, height];
        # lf_crop no shrinkage: &lf_crop [[0, null], [133, 1121], [323, 1596]]  # [133, 1121], [323, 1596] / 19 = [7, 59], [17, 84] # dim in lenselets: 70, 85
        crop_llets_each_float = shrink / scale
        crop_llets_each = int(crop_llets_each_float)
        assert crop_llets_each == crop_llets_each_float, (crop_llets_each, crop_llets_each_float)
        max_llet_crops = [70, 85]
        crop = [
            [max(0, bc[0] - crop_llets_each * nnum), min(mlc, bc[1] // nnum + crop_llets_each) * nnum]
            for bc, mlc in zip(base_crop, max_llet_crops)
        ]
    else:
        raise NotImplementedError(crop_name)

    return [[0, None]] + crop  # add crop for channel


def get_crops(
    affine_trf_name: str, lf_crop: Optional[Sequence[Sequence[Optional[int]]]], meta: dict, for_slice: bool = False
) -> Tuple[Sequence[Sequence[Optional[int]]], Sequence[Sequence[Optional[int]]], Sequence[Sequence[Optional[int]]]]:

    scale: int = meta["scale"]
    nnum: int = meta["nnum"]
    z_ls_rescaled: int = meta.get("z_ls_rescaled", 0)
    assert for_slice or z_ls_rescaled, "invalid (=0) or missing 'z_ls_rescaled' in meta"
    shrink: int = meta["shrink"]


    ref_crop_in = [[meta["pred_z_min"], meta["pred_z_max"]]] + get_lf_crop(
        affine_trf_name, shrink=shrink, nnum=nnum, scale=scale
    )[1:]

    if lf_crop is not None:
        assert len(lf_crop) == 3, "cyx"
        assert lf_crop[0][0] == 0, "dont crop lf channel (here treated as z)"
        assert lf_crop[0][1] is None, "dont crop lf channel (here treated as z)"
        assert all([len(lfc) == 2 for lfc in lf_crop])
        lf_crop = [list(lfc) for lfc in lf_crop]
        for i, lfc in enumerate(lf_crop):
            ref_crop_in[i][0] += lfc[0]
            if lfc[1] is not None:
                if lfc[1] >= 0:
                    lfc[1] = lfc[1] - get_lf_shape(affine_trf_name)[i-1]

                assert lfc[1] < 0

                ref_crop_in[i][1] += lfc[1]

    if ref_crop_in == [[0, 838], [0, None], [0, None]]:
        ls_crop = [[0, None]] * 3
    elif affine_trf_name == "Heart_tightCrop" and ref_crop_in in [
        [[0, 838], [57, -57], [57, -57]],
        [[0, 838], [57, 1273-57], [57, 1463-57]]]:
        ls_crop = (19, -10), (152, -133), (171, -133)
    elif affine_trf_name == "Heart_tightCrop" and ref_crop_in in [
        [[0, 838], [38, -38], [38, -38]],
        [[0, 838], [38, 1273-38], [38, 1463-38]],
    ]:
        ls_crop = (18, -12), (57 + 3 * 19, -57 - 2 * 19), (95 + 19, -57 - 19)
    elif affine_trf_name == "Heart_tightCrop" and ref_crop_in in [
        [[0, 838], [19, -19], [19, -19]],
        [[0, 838], [19, 1273-19], [19, 1463-19]],   #  [19, 1235], [19, 1425
    ]:
        ls_crop = (18, -12), (57 + 3 * 19, -57 - 2 * 19), (95 + 19, -57 - 19)
    elif affine_trf_name == "wholeFOV" and ref_crop_in in [
        [[0, 838], [57, -57], [57, -57]],
        [[0, 838], [57, 1064 - 57], [57, 1083 - 57]],
    ]:
        ls_crop = (
            (19, -13),
            (114 + 4 * 19, -114 - 4 * 19),
            (133 + 3 * 19, -95 - 3 * 19),
        )
    elif affine_trf_name == "wholeFOV" and ref_crop_in in [
        [[0, 838], [38, -38], [38, -38]],
        [[0, 838], [38, 1064 - 38], [38, 1083 - 38]],
    ]:
        ls_crop = (
            (19, -13),
            (114 + 4 * 19, -114 - 4 * 19),
            (133 + 3 * 19, -95 - 3 * 19),
        )
    elif affine_trf_name == "wholeFOV" and ref_crop_in in [
        [[0, 838], [19, -19], [19, -19]],
        [[0, 838], [19, 1064 - 19], [19, 1083 - 19]],
    ]:
        ls_crop = (
            (19, -13),
            (114 + 3 * 19, -114 - 3 * 19),
            (133 + 2 * 19, -95 - 2 * 19),
        )
    elif affine_trf_name == "gcamp" and ref_crop_in == [[152, 615], [76, 1178], [266, 1615]]:
        assert scale == 2
        assert shrink == 6
        ls_crop = (60, 181), (16, -36), (46, -10)  # f2
    elif affine_trf_name == "gcamp" and ref_crop_in == [[152, 615], [95, 1159], [285, 1615]]:
        assert scale == 4
        assert shrink == 8
        ls_crop = (60, 181), (32, -72), (92, -20)  # f4
    elif affine_trf_name == "gcamp" and ref_crop_in == [[152, 615], [114, 1140], [304, 1615]]:
        assert scale == 8
        assert shrink == 8
        ls_crop = (60, 181), (64, -144), (184, -40)  # f8
    else:
        raise NotImplementedError((affine_trf_name, ref_crop_in))

    mini_crop = get_raw_ls_crop(affine_trf_name, wrt_ref=True)
    lower = [mc[0] + a[0] for mc, a in zip(mini_crop, ls_crop)]
    upper = [
        None if mc[1] is None and a[1] is None else mc[1] if a[1] is None else a[1] if mc[1] is None else a[1] + mc[1]
        for mc, a in zip(mini_crop, ls_crop)
    ]

    ref_crop_out = [(low, up) for low, up in zip(lower, upper)]

    if for_slice:
        scale = meta.get("ls_slice_scale", scale)

    ls_crop_float = [
        [None if zc is None else zc * z_ls_rescaled / get_ls_shape(affine_trf_name)[0] for zc in ls_crop[0]]
    ] + [(lsc[0] / nnum * scale, None if lsc[1] is None else lsc[1] / nnum * scale) for lsc in ls_crop[1:]]
    ls_crop = [
        [None if zc is None else zc * z_ls_rescaled // get_ls_shape(affine_trf_name)[0] for zc in ls_crop[0]]
    ] + [
        (round(lsc[0] / nnum * scale), None if lsc[1] is None else round(lsc[1] / nnum * scale)) for lsc in ls_crop[1:]
    ]
    assert len(ls_crop) == 3
    if for_slice:
        ls_crop_float[0] = [0, None]
        ls_crop[0] = [0, None]

    if ls_crop_float != ls_crop:
        warnings.warn(f"rounded ls crop from {ls_crop_float} to {ls_crop}")

    return ref_crop_in, ref_crop_out, [[0, None]] + ls_crop


if __name__ == "__main__":
    pass
