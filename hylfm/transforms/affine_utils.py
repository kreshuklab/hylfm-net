import logging
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

MAX_SRHINK_IN_LENSLETS = 3

Heart_tightCrop = "Heart_tightCrop"
staticHeartFOV = "staticHeartFOV"
wholeFOV = "wholeFOV"
fast_cropped_8ms = "fast_cropped_8ms"
fast_cropped_6ms = "fast_cropped_6ms"
gcamp = "gcamp"
heart_2020_02_fish1_static = "heart_2020_02_fish1_static"
heart_2020_02_fish2_static = "heart_2020_02_fish2_static"


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
    elif name == Heart_tightCrop:
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
    elif name == staticHeartFOV:
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
    elif name == wholeFOV:
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
    elif name == fast_cropped_8ms:
        return [
            [
                0.9986010487421885,
                -0.002120810141566837,
                -0.008149618765723362,
                4.753590473932092,
                -7.497126762641354e-4,
                0.999739497335616,
                -0.002100267232562155,
                1.5264448905391126,
                0.012767265132301005,
                0.008313927705739728,
                1.0251251343976073,
                -23.383246688659526,
            ],
            [
                1.0014699495221129,
                0.0022217885570053057,
                0.007815338643888291,
                -4.714470966958904,
                0.0010303617894547048,
                1.0000489406185478,
                0.001449929955455748,
                -1.2415661154926954,
                -0.012897434787601393,
                -0.008282360222404195,
                0.9728093345347644,
                24.192111553605027,
            ],
            [
                0.9999756097799303,
                -3.14068807737589e-5,
                2.6414434466927684e-6,
                0.03299776828073804,
                -7.443462885587118e-6,
                1.0000205503425315,
                6.372773046833699e-5,
                -0.032715113781997054,
                3.53248611190156e-5,
                2.2833262680210066e-5,
                1.000242859147333,
                -0.1291290449835767,
            ],
            [
                0.9997562287077025,
                -1.4690002895491214e-4,
                -1.3886713774554479e-5,
                0.23433800490792453,
                -2.4140517872558393e-4,
                1.0002056303267557,
                7.243447875154269e-4,
                -0.3313980780587243,
                3.9422039654427526e-4,
                1.433164031456372e-4,
                1.0024346602689587,
                -1.2459354397517026,
            ],
            [
                0.9977154744301842,
                -0.002241191411341819,
                -0.009852533063965996,
                6.0748338797101615,
                -2.1704150642624007e-4,
                1.000024837549752,
                -1.479204794364181e-4,
                0.35024621525122906,
                0.013231836246675045,
                0.008248997222915135,
                1.025443845475132,
                -23.83731141291179,
            ],
            [
                1.002301101653545,
                0.001244721528787248,
                0.006274675516564312,
                -4.134709347642092,
                2.737459409621002e-4,
                1.00055221854111,
                0.0016732066436690894,
                -1.2089095422920313,
                -0.013232668769306219,
                -0.008141878604109042,
                0.9701538227526025,
                25.280673996062564,
            ],
            [
                1.0036672920493648,
                -0.001217988458688357,
                -0.029144491141293156,
                45.37683487750689,
                -1.6696390203153297e-4,
                1.003610058305277,
                0.011245829556812224,
                95.22415193683851,
                0.010100024209689917,
                -0.005478727449094312,
                0.9926932232933254,
                7.776297843932661,
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
    elif name == fast_cropped_6ms:
        return [
            [
                1.0002459794124785,
                -2.2800776616194434e-4,
                7.06635354005253e-4,
                -0.310507600323082,
                -1.6436277044339917e-4,
                1.000130416649231,
                1.2675415989102842e-4,
                0.02071198230079377,
                -0.006819804116110938,
                0.002760002365515835,
                1.005591969344203,
                1.3516919745339329,
            ],
            [
                0.9997554273413035,
                2.3605194775509156e-4,
                -6.924948788607131e-4,
                0.30161248423846176,
                1.7307578986511582e-4,
                0.9998857576307465,
                -1.2431688124025126e-4,
                -0.033720084165848305,
                0.006789030910848358,
                -0.002745119133451576,
                0.9944079280754539,
                -1.336539711845158,
            ],
            [
                1.0002999931827274,
                -1.7424955178460828e-4,
                7.747523582806886e-4,
                -0.3886584055398829,
                -7.526945519961514e-5,
                1.0002724045539166,
                1.2600866536445355e-4,
                -0.09406334524547129,
                -0.0067096819112248745,
                0.0027545563116734037,
                1.0053461443220024,
                1.3787048653452845,
            ],
            [
                1.0003045576351692,
                0.0013508434831580461,
                -1.6493465412205095e-4,
                -0.7432324164894379,
                4.845596049328561e-4,
                1.0013082528497923,
                9.678886678765577e-4,
                -1.408435427994669,
                0.008001777014911319,
                -0.0038694726723087925,
                0.9920015459459663,
                -0.543309646302159,
            ],
            [
                0.9995007975492034,
                -0.001160436056024711,
                -4.656763320559507e-4,
                1.0034378137190076,
                -3.070602992248084e-4,
                0.9984231547086668,
                -9.998896739684166e-4,
                1.4002372648270065,
                -0.0028081724905490166,
                9.91407312774693e-4,
                1.0020357549233985,
                0.40823789022980733,
            ],
            [
                1.0001673226422105,
                -2.5007202083074224e-4,
                2.3942102071763976e-4,
                -0.08197644365428648,
                -3.0761646270283387e-4,
                1.000149108002461,
                2.6453062735372666e-4,
                0.05459382373050764,
                -0.005646233706300736,
                0.002997753393853305,
                1.006828811526157,
                0.032925484769386906,
            ],
            [
                1.0000641426395687,
                3.991412305563949e-5,
                3.8494220770142213e-4,
                -0.18681276543478734,
                3.7993739239118724e-5,
                1.0002093325561061,
                -1.7142164015812985e-5,
                -0.11247435589269912,
                3.9713356658404464e-4,
                -4.876427889456174e-5,
                0.9992308054006369,
                0.07106374786050755,
            ],
            [
                1.0006635981556136,
                3.2484007182119126e-4,
                -0.0013487090395483512,
                -0.19023673737013777,
                4.5599607110634517e-4,
                1.0020959354993126,
                0.002001401251431046,
                -2.0316561939875983,
                0.009072407722233547,
                -0.002635303624325553,
                0.9923448975374671,
                -1.5472322100460143,
            ],
            [
                0.9993290054471409,
                2.2427309163849024e-4,
                6.354544506468878e-4,
                0.0707471753866127,
                -1.0642995073706009e-4,
                0.999778393862275,
                9.938278259580958e-4,
                -0.14354387390236484,
                -0.003938578709303977,
                0.003172926018743711,
                1.0012307830969316,
                0.09969919601110715,
            ],
            [
                0.9987285932863739,
                1.5557144455591472e-4,
                -0.013691022603613988,
                120.6883922598818,
                -4.526758033175239e-4,
                1.000082580133885,
                0.0019164519758949493,
                88.0206519687398,
                0.007785551759521633,
                0.006814207789026994,
                1.0095319730590382,
                -20.111755129095247,
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
    elif name == gcamp:
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
    elif name == heart_2020_02_fish1_static:
        return [
            [
                0.98115,
                0.0076884,
                -0.16868,
                -37.8586,
                -0.011549,
                0.98008,
                -0.70569,
                -3.7397,
                0.027099,
                0.013045,
                3.9761,
                -89.4007,
            ]
        ]
    else:
        raise NotImplementedError(name)


def get_raw_lf_shape(crop_name: str, *, wrt_ref: bool, nnum: int = None, scale: int = None) -> List[int]:
    if crop_name == Heart_tightCrop:
        ref_shape = [1273, 1463]  # crop on raw (1551,1351) in matlab: [250,300,1550,1350]
    elif crop_name == staticHeartFOV:
        ref_shape = [
            1178,
            1767,
        ]  # crop on raw in matlab: rect_LF = [100, 400, 1850, 1250]; %[xmin, ymin, width, height]
    elif crop_name == wholeFOV:
        ref_shape = [
            1064,
            1083,
        ]  # crop on raw in matlab: rect_LF = [450, 450, 1150, 1150]; %[xmin, ymin, width, height];
    elif crop_name == fast_cropped_8ms:
        ref_shape = [1273, 1254]  # crop on raw in matlab: rect_LF = [0, 0, 1350, 1350]; %[xmin, ymin, width, height];
    elif crop_name == fast_cropped_6ms:
        ref_shape = [931, 931]  # [0,0,1024,1024]
    elif crop_name == gcamp:
        # crop on raw in matlab: rect_LF = [174, 324, 1700, 1400]; %[250, 300, 1550, 1350]; %[xmin, ymin, width, height];
        ref_shape = [1330, 1615]
    elif crop_name == heart_2020_02_fish1_static:
        #     # rect_LF = [350, 350, 1300, 1300]; %[xmin, ymin, width, height];
        #     # rect_LS = [300, 300, 1400, 1400];
        ref_shape = [1235, 1235]
    elif crop_name == heart_2020_02_fish2_static:
        ref_shape = [1140, 1520]
    else:
        raise NotImplementedError(crop_name)

    if wrt_ref:
        shape = ref_shape
    else:
        if nnum is None:
            raise TypeError("nnum missing")

        if scale is None:
            raise TypeError("expected_scale missing")

        shape_float = [rs / nnum * scale for rs in ref_shape]
        shape = [int(s) for s in shape_float]
        assert shape == shape_float, (shape, shape_float)

    return shape


get_precropped_lf_shape = get_raw_lf_shape


def get_lf_roi_in_raw_lf(
    crop_name: str, *, shrink: int, nnum: int, scale: int, wrt_ref: bool
) -> Tuple[Tuple[int, int], ...]:
    crop_llets_each_float = MAX_SRHINK_IN_LENSLETS - shrink / scale
    crop_llets_each = int(crop_llets_each_float)
    assert crop_llets_each >= 0
    assert crop_llets_each == crop_llets_each_float, (crop_llets_each, crop_llets_each_float)

    upscale = nnum if wrt_ref else scale
    return tuple(
        (crop_llets_each * upscale, s - crop_llets_each * upscale)
        for s in get_raw_lf_shape(crop_name, nnum=nnum, scale=scale, wrt_ref=wrt_ref)
    )


get_lf_roi_in_precropped_lf = get_lf_roi_in_raw_lf


def get_lf_shape(crop_name: str, *, shrink: int, nnum: int, scale: int, wrt_ref: bool):
    return [
        r[1] - r[0]
        for r in get_lf_roi_in_precropped_lf(crop_name, shrink=shrink, nnum=nnum, scale=scale, wrt_ref=wrt_ref)
    ]


def get_pred_roi_in_precropped_lf(
    crop_name: str, *, shrink: int, nnum: int, scale: int, wrt_ref: bool
) -> List[List[int]]:
    shrink_llnets_float = shrink / scale
    shrink_llnets = int(shrink_llnets_float)
    assert shrink_llnets >= 0
    assert shrink_llnets == shrink_llnets_float, (shrink_llnets, shrink_llnets_float)

    upscale = nnum if wrt_ref else scale
    return [
        [r[0] + shrink_llnets * upscale, r[1] - shrink_llnets * upscale]
        for r in get_lf_roi_in_precropped_lf(crop_name, nnum=nnum, scale=scale, shrink=shrink, wrt_ref=wrt_ref)
    ]


def get_pred_shape(crop_name: str, *, shrink: int, nnum: int, scale: int, wrt_ref: bool, z_out: int):
    return [z_out] + [
        r[1] - r[0]
        for r in get_pred_roi_in_precropped_lf(crop_name, shrink=shrink, nnum=nnum, scale=scale, wrt_ref=wrt_ref)
    ]


def get_ls_ref_shape(crop_name: str):
    if crop_name == Heart_tightCrop:
        # crop in matlab: 200, 250, 1650, 1450 for ref shape
        # crop for ref shape + crop for divisibility
        ref_roi = [[0, 241], [0, 1700 - 249 - 0], [0, 1850 - 199 - 0]]
    elif crop_name == staticHeartFOV:
        # crop in matlab: 50, 300, 1950, 1450 for ref shape
        # crop for ref shape + crop for divisibility
        ref_roi = [[0, 241], [0, 1750 - 299 - 0], [0, 2000 - 49 - 0]]
    elif crop_name == wholeFOV:
        # crop in matlab: # rect_LS = [350, 350, 1350, 1350]; %[xmin, ymin, width, height];
        # crop for ref shape + crop for divisibility
        ref_roi = [[0, 241], [0, 1700 - 349 - 0], [0, 1700 - 349 - 0]]
    elif crop_name == fast_cropped_8ms:
        # crop in matlab: # rect_LS = [0, 0, 1350, 1350]; %[xmin, ymin, width, height];
        # crop for ref shape + crop for divisibility
        ref_roi = [[0, 241], [0, 1350 + 1 - 0], [0, 1350 + 1 - 0]]
    elif crop_name == fast_cropped_6ms:
        # crop in matlab: # rect_LS = [0, 0, 1024, 1024]; %[xmin, ymin, width, height];
        # crop for ref shape + crop for divisibility
        ref_roi = [[0, 241], [0, 1024 + 1 - 0], [0, 1024 + 1 - 0]]
    elif crop_name == gcamp:
        # in matlab: 124, 274, 1800, 1500
        ref_roi = [[0, 241], [0, 1774 - 273], [0, 1924 - 123 - 0]]
    elif crop_name == heart_2020_02_fish1_static:
        # rect_LF = [350, 350, 1300, 1300]; %[xmin, ymin, width, height];
        # rect_LS = [300, 300, 1400, 1400];
        ref_roi = [[0, 241], [0, 1401], [0, 1401]]
    elif crop_name == heart_2020_02_fish2_static:
        # rect_LF = [230, 330, 1600, 1200]; %[xmin, ymin, width, height];
        # rect_LS = [180, 280, 1700, 1300]; %[xmin, ymin, width, height];
        ref_roi = [[0, 241], [0, 1301], [0, 1701]]
    else:
        raise NotImplementedError(crop_name)

    return [r[1] - r[0] for r in ref_roi]


def get_precropped_ls_roi_in_raw_ls(crop_name: str, *, for_slice: bool, wrt_ref: bool) -> List[List[Optional[int]]]:
    if wrt_ref:
        if crop_name == Heart_tightCrop:
            # crop in matlab: 200, 250, 1650, 1450 for ref shape
            # crop for ref shape + crop for divisibility
            precropped_roi = [[0, 241], [3, 1700 - 249 - 4], [8, 1850 - 199 - 9]]
        elif crop_name == staticHeartFOV:
            # crop in matlab: 50, 300, 1950, 1450 for ref shape
            # crop for ref shape + crop for divisibility
            precropped_roi = [[0, 241], [3, 1750 - 299 - 4], [6, 2000 - 49 - 7]]
        elif crop_name == wholeFOV:
            # crop in matlab: # rect_LS = [350, 350, 1350, 1350]; %[xmin, ymin, width, height];
            # crop for ref shape + crop for divisibility
            precropped_roi = [[0, 241], [1, 1700 - 349 - 1], [1, 1700 - 349 - 1]]
        elif crop_name == fast_cropped_8ms:
            # crop in matlab: # rect_LS = [0, 0, 1350, 1350]; %[xmin, ymin, width, height];
            # crop for ref shape + crop for divisibility
            precropped_roi = [[0, 241], [1, 1350 + 1 - 1], [1, 1350 + 1 - 1]]
        elif crop_name == fast_cropped_6ms:
            # crop in matlab: # rect_LS = [0, 0, 1024, 1024]; %[xmin, ymin, width, height];
            # crop for ref shape + crop for divisibility
            precropped_roi = [[0, 241], [9, 1024 + 1 - 9], [9, 1024 + 1 - 9]]
        elif crop_name == gcamp:
            # in matlab: 124, 274, 1800, 1500
            precropped_roi = [[0, 241], [0, 1774 - 273], [7, 1924 - 123 - 8]]
        elif crop_name == heart_2020_02_fish1_static:
            # rect_LF = [350, 350, 1300, 1300]; %[xmin, ymin, width, height];
            # rect_LS = [300, 300, 1400, 1400];
            precropped_roi = [[0, 241], [7, 1401 - 7], [7, 1401 - 7]]
        elif crop_name == heart_2020_02_fish2_static:
            # rect_LS = [180, 280, 1700, 1300]; %[xmin, ymin, width, height];
            precropped_roi = [[0, 241], [4, 1301 - 5], [5, 1701 - 5]]
        else:
            raise NotImplementedError(crop_name)
    else:
        if crop_name == Heart_tightCrop:
            # crop in matlab: 200, 250, 1650, 1450 for ref shape
            # crop for ref shape + crop for divisibility
            precropped_roi = [[0, 241], [249 + 3, 1700 - 4], [199 + 8, 1850 - 9]]
        elif crop_name == staticHeartFOV:
            # crop in matlab: 50, 300, 1950, 1450 for ref shape
            # crop for ref shape + crop for divisibility
            precropped_roi = [[0, 241], [299 + 3, 1750 - 4], [49 + 6, 2000 - 7]]
        elif crop_name == wholeFOV:
            # crop in matlab: # rect_LS = [350, 350, 1350, 1350]; %[xmin, ymin, width, height];
            # crop for ref shape + crop for divisibility
            precropped_roi = [[0, 241], [349 + 1, 1700 - 1], [349 + 1, 1700 - 1]]
        elif crop_name == fast_cropped_8ms:
            # crop in matlab: # rect_LS = [0, 0, 1350, 1350]; %[xmin, ymin, width, height];
            # crop for ref shape + crop for divisibility
            precropped_roi = [[0, 241], [0, 1350 - 1], [0, 1350 - 1]]
        elif crop_name == fast_cropped_6ms:
            # crop in matlab: # rect_LS = [0, 0, 1024, 1024]; %[xmin, ymin, width, height];
            # crop for ref shape + crop for divisibility
            precropped_roi = [[0, 241], [9, 1024 + 1 - 9], [9, 1024 + 1 - 9]]
        elif crop_name == gcamp:
            precropped_roi = [[0, 241], [273, 1774], [123 + 7, 1924 - 8]]
        elif crop_name == heart_2020_02_fish1_static:
            # rect_LF = [350, 350, 1300, 1300]; %[xmin, ymin, width, height];
            # rect_LS = [300, 300, 1400, 1400];
            precropped_roi = [[0, 241], [299 + 7, 299 + 1401 - 7], [299 + 7, 299 + 1401 - 7]]
        elif crop_name == heart_2020_02_fish2_static:
            # rect_LS = [180, 280, 1700, 1300]; %[xmin, ymin, width, height];
            precropped_roi = [[0, 241], [279 + 4, 279 + 1301 - 5], [179 + 5, 179 + 1701 - 5]]
        else:
            raise NotImplementedError(crop_name)

    if for_slice:
        precropped_roi[0] = [0, 1]

    return precropped_roi


def get_precropped_ls_shape(
    crop_name: str, *, nnum: int, for_slice: bool, wrt_ref: bool, ls_scale: int = None
) -> List[int]:
    precropped_ls_shape_float = [
        r[1] - r[0] for r in get_precropped_ls_roi_in_raw_ls(crop_name, for_slice=for_slice, wrt_ref=wrt_ref)
    ]
    precropped_ls_shape_float[1] /= nnum
    precropped_ls_shape_float[2] /= nnum
    precropped_ls_shape = [int(s) for s in precropped_ls_shape_float]
    if precropped_ls_shape != precropped_ls_shape_float:
        logger.warning("rounding precropped_ls_shape: %s, %s", precropped_ls_shape, precropped_ls_shape_float)

    if not wrt_ref and ls_scale is None:
        raise TypeError("ls_scale required if not 'wrt_ref'")

    upscale = nnum if wrt_ref else ls_scale
    precropped_ls_shape[1] *= upscale
    precropped_ls_shape[2] *= upscale
    return precropped_ls_shape


def get_ls_roi(  # either in ref vol or in precroped ls
    crop_name: str, *, for_slice: bool, nnum: int, wrt_ref: bool, z_ls_rescaled: int, ls_scale: int
) -> Tuple[Tuple[int, int], ...]:
    if crop_name == Heart_tightCrop:
        ls_crop = [[19, 13], [7 * 19, 6 * 19], [7 * 19, 5 * 19]]
    elif crop_name == wholeFOV:
        ls_crop = [[19, 13], [10 * 19, 10 * 19], [10 * 19, 8 * 19]]
    elif crop_name == staticHeartFOV:  # todo: check numbers, this is just copied form 'wholeFOV'
        ls_crop = [[19, 13], [10 * 19, 10 * 19], [10 * 19, 8 * 19]]
    elif crop_name == fast_cropped_8ms:  # todo: check numbers, this is just a guess
        ls_crop = [[19, 13], [4 * 19, 3 * 19], [4 * 19, 2 * 19]]
    elif crop_name == fast_cropped_6ms:  # todo: check numbers, this is just a guess
        ls_crop = [[19, 13], [4 * 19, 3 * 19], [4 * 19, 2 * 19]]
    elif crop_name == gcamp:
        ls_crop = [[60, 60], [4 * 19, 9 * 19], [9 * 19, 6 * 19]]
    elif crop_name == heart_2020_02_fish1_static:
        ls_crop = [[16, 16], [5 * 19, 5 * 19], [5 * 19, 5 * 19]]
    elif crop_name == heart_2020_02_fish2_static:
        ls_crop = [[16, 16], [5 * 19, 5 * 19], [5 * 19, 5 * 19]]
    else:
        raise NotImplementedError(crop_name)

    assert all([rr >= 0 for r in ls_crop for rr in r])
    assert all([rr % nnum == 0 for r in ls_crop[1:] for rr in r]), ls_crop

    if for_slice:
        ls_crop[0] = [0, 0]

    if z_ls_rescaled != 241:
        raise NotImplementedError(z_ls_rescaled)

    if wrt_ref:
        ls_roi = get_precropped_ls_roi_in_raw_ls(crop_name, for_slice=for_slice, wrt_ref=True)
        assert all([rr >= 0 for r in ls_roi for rr in r]), ls_roi
    else:
        ls_roi = [
            [0, s]
            for s in get_precropped_ls_shape(
                crop_name, for_slice=for_slice, wrt_ref=False, nnum=nnum, ls_scale=ls_scale
            )
        ]
        ls_crop = ls_crop[:1] + [[cc // nnum * ls_scale for cc in c] for c in ls_crop[1:]]

    return tuple((r[0] + c[0], r[1] - c[1]) for r, c in zip(ls_roi, ls_crop))


def get_ls_shape(
    crop_name: str,
    *,
    pred_z_min: int,
    pred_z_max: int,
    for_slice: bool,
    shrink: int,
    scale: int,
    nnum: int,
    z_ls_rescaled: int,
    ls_scale: int,
):
    return [
        r[1] - r[0]
        for r in get_ls_roi(
            crop_name,
            pred_z_min=pred_z_min,
            pred_z_max=pred_z_max,
            for_slice=for_slice,
            scale=scale,
            nnum=nnum,
            wrt_ref=False,
            z_ls_rescaled=z_ls_rescaled,
            ls_scale=ls_scale,
            shrink=shrink,
        )
    ]


if __name__ == "__main__":
    pass
