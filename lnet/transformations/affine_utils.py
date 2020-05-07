from typing import List, Tuple, Optional


def get_bdv_affine_transformations_by_name(name: str) -> List[List[float]]:
    if name == "Heart_tightCrop":
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
    else:
        raise NotImplementedError(name)

def get_ref_crop_out(affine_trf_name: str, ref_crop_in: Tuple[
    Tuple[int, Optional[int]], Tuple[int, Optional[int]], Tuple[int, Optional[int]]], inverted: bool) -> Tuple[
    Tuple[int, Optional[int]], Tuple[int, Optional[int]], Tuple[int, Optional[int]]]:
    if inverted:
        raise NotImplementedError

    if affine_trf_name == "Heart_tightCrop" and ref_crop_in == ((0, 838), (57, -57), (57, -57)):
        return ((19, -10), (152, -133), (171, -133))
    else:
        raise NotImplementedError((affine_trf_name, ref_crop_in))
