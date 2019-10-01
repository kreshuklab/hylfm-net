from pathlib import Path

from lnet.dataset_configs import DatasetConfigEntry


class Fish00(DatasetConfigEntry):
    common_path = Path(
        "/g/hufnagel/LF/NNG/Experiments/Olympus20x0.5NA/301018_Jakob_sparseLabeling/ISce_p2890/fish1/140Hz6msDelay/wholeheart"
    )
    description = "blood cells"


fish00_0_left = Fish00("2018-10-30_04.42.19/stack_1_channel_7", "Rectified_LC", "LCout")
fish00_0_right = Fish00("2018-10-30_04.42.19/stack_1_channel_7", "Rectified_RC", "RCout")
fish00_1_left = Fish00("2018-10-30_04.45.58/stack_1_channel_7", "Rectified_LC", "LCout")
fish00_1_right = Fish00("2018-10-30_04.45.58/stack_1_channel_7", "Rectified_RC", "RCout")
fish00_2_left = Fish00("2018-10-30_04.46.32/stack_1_channel_7", "Rectified_LC", "LCout")
fish00_2_right = Fish00("2018-10-30_04.46.32/stack_1_channel_7", "Rectified_RC", "RCout")
fish00_3_left = Fish00("2018-10-30_04.51.25/stack_1_channel_7", "Rectified_LC", "LCout")
fish00_3_right = Fish00("2018-10-30_04.51.25/stack_1_channel_7", "Rectified_RC", "RCout")
fish00_4_left = Fish00("2018-10-30_04.52.22/stack_1_channel_7", "Rectified_LC", "LCout")
fish00_4_right = Fish00("2018-10-30_04.52.22/stack_1_channel_7", "Rectified_RC", "RCout")
fish00_5_left = Fish00("2018-10-30_04.53.21/stack_1_channel_7", "Rectified_LC", "LCout")
fish00_5_right = Fish00("2018-10-30_04.53.21/stack_1_channel_7", "Rectified_RC", "RCout")
fish00_6_left = Fish00("2018-10-30_04.54.41_veryNice/stack_1_channel_7", "Rectified_LC", "LCout")
fish00_6_right = Fish00("2018-10-30_04.54.41_veryNice/stack_1_channel_7", "Rectified_RC", "RCout")
fish00_7_left = Fish00("2018-10-30_04.57.07_bothNice/stack_1_channel_7", "Rectified_LC", "LCout")
fish00_7_right = Fish00("2018-10-30_04.57.07_bothNice/stack_1_channel_7", "Rectified_RC", "RCout")
fish00_8_left = Fish00("2018-10-30_05.43.47_promising/stack_1_channel_7", "Rectified_LC", "LCout")
fish00_8_right = Fish00("2018-10-30_05.43.47_promising/stack_1_channel_7", "Rectified_RC", "RCout")
fish00_9_left = Fish00("2018-10-30_05.48.20_100Laser/stack_1_channel_7", "Rectified_LC", "LCout")
fish00_9_right = Fish00("2018-10-30_05.48.20_100Laser/stack_1_channel_7", "Rectified_RC", "RCout")


class Fish01(DatasetConfigEntry):
    common_path = Path(
        "/g/hufnagel/LF/NNG/Experiments/Olympus20x0.5NA/060918_Jakob/7943x7943/fish1/140Hz_6msDelay/wholeHeart"
    )
    description = "heart muscle"


fish01_0_left = Fish01("fov10/2018-09-06_13.13.16/stack_1_channel_0", "Rectified_LC", "LCout")
fish01_0_right = Fish01("fov10/2018-09-06_13.13.16/stack_1_channel_0", "Rectified_RC", "RCout")  # access right misssing
fish01_1_left = Fish01("fov1/2018-09-06_12.24.20/stack_1_channel_0", "Rectified_LC", "LCout")
fish01_1_right = Fish01("fov1/2018-09-06_12.24.20/stack_1_channel_0", "Rectified_RC", "RCout")  # access right missing
fish01_2_left = Fish01("fov3/2018-09-06_12.34.43/stack_1_channel_0", "Rectified_LC", "LCout")  # Rectified_LC emtpy
fish01_2_right = Fish01("fov3/2018-09-06_12.34.43/stack_1_channel_0", "Rectified_RC", "RCout")  # access right missing
fish01_3_left = Fish01("fov3/2018-09-06_12.36.03/stack_1_channel_0", "Rectified_LC", "LCout")  # Rectified_LC emtpy
fish01_3_right = Fish01("fov3/2018-09-06_12.36.03/stack_1_channel_0", "Rectified_RC", "RCout")  # access right missing
fish01_4_left = Fish01("fov4/2018-09-06_12.38.34/stack_1_channel_0", "Rectified_LC", "LCout")
fish01_4_right = Fish01("fov4/2018-09-06_12.38.34/stack_1_channel_0", "Rectified_RC", "RCout")  # access right missing
fish01_5_left = Fish01("fov6/2018-09-06_12.42.19/stack_1_channel_0", "Rectified_LC", "LCout")
fish01_5_right = Fish01("fov6/2018-09-06_12.42.19/stack_1_channel_0", "Rectified_RC", "RCout")  # access right missing
fish01_6_left = Fish01("fov7/2018-09-06_12.47.10/stack_1_channel_0", "Rectified_LC", "LCout")
fish01_6_right = Fish01("fov7/2018-09-06_12.47.10/stack_1_channel_0", "Rectified_RC", "RCout")  # access right misssing
# sort of 'missing' duplicate of fish01_8
# fish01_7_left = Fish01(
#     "fov8_promising/2018-09-06_12.58.29/stack_1_channel_0", "LC/Rect", "LCout1"
# )  # LC/Rect not existing
# fish01_7_right = Fish01(
#     "fov8_promising/2018-09-06_12.58.29/stack_1_channel_0", "RC/Rect", "RCout1"
# )  # RC/Rect not existing
fish01_8_left = Fish01("fov8_promising/2018-09-06_12.58.29/stack_1_channel_0", "Rectified_LC", "LCout")
fish01_8_right = Fish01(
    "fov8_promising/2018-09-06_12.58.29/stack_1_channel_0", "Rectified_RC", "RCout"
)  # access right missing
fish01_9_left = Fish01(
    "fov9/2018-09-06_13.11.15/stack_1_channel_0", "Rectified_LC", "LCout"
)  # missing samples (empty tif files)
fish01_9_right = Fish01(
    "fov9/2018-09-06_13.11.15/stack_1_channel_0", "Rectified_RC", "RCout"
)  # no folder 'Rectified_RC'


class Fish02(DatasetConfigEntry):
    common_path = Path("/g/hufnagel/LF/LenseLeNet_Microscope/20190605_Fish/8214x7943_H2A-mC_H2B-EGFP/fish2_arrythmic/Fish_120to120_241planes")
    description = "F02"


fish02_LS0 = Fish02(
    "20190607/fish3_stable/2019-06-07_04.02.02_30tp/stack_6_channel_0",
    "lfimg",
    "gt/LS_Stack",
    "FirstStaticFishLS_561nm",
    x_roi=(slice(38, None), slice(None)),
    y_roi=(slice(None), slice(38, None), slice(None)),
)

fish02_LS0_filet = Fish02(
    "20190607/fish3_stable/2019-06-07_04.02.02_30tp/stack_6_channel_0",
    "lfimg",
    "gt/LS_Stack",
    "FirstStaticFishLS_561nm",
    x_roi=(slice(38, None), slice(None)),
    y_roi=(slice(7, 42), slice(38, None), slice(None)),
)

fish02_LF0 = Fish02(
    "20190607/fish3_stable/2019-06-07_04.02.02_30tp/stack_6_channel_0",
    "lfimg",
    "gt/LF_Recon",
    "FirstStaticFishLF_561nm",
    x_roi=(slice(38, None), slice(None)),
    y_roi=(slice(None), slice(38, None), slice(None)),
)
fish02_LS1 = Fish02(
    "20190614_Jakob/7943x8214/488LS_488LF_BSdetection/fish2_static",
    "lfimg",
    "gt/LS_Stack",
    "FishStaticLS_488nm",
    x_roi=(slice(None), slice(1, None)),
    y_roi=(slice(None), slice(None), slice(1, None)),
)
fish02_LF1 = Fish02(
    "20190614_Jakob/7943x8214/488LS_488LF_BSdetection/fish2_static",
    "lfimg",
    "gt/LF_Stack",
    "FishStaticLF_488nm",
    x_roi=(slice(None), slice(1, None)),
    y_roi=(slice(None), slice(None), slice(1, None)),
)
fish02_LF1_crop0 = Fish02(
    "20190614_Jakob/7943x8214/488LS_488LF_BSdetection/fish2_static",
    "lfimg",
    "gt/LF_Stack",
    "FishStaticLF_488nm_crop0",
    x_roi=(slice(19, -19), slice(39, -19)),
    y_roi=(slice(None), slice(19, -19), slice(39, -19)),
)
fish02_LF1_crop1 = Fish02(
    "20190614_Jakob/7943x8214/488LS_488LF_BSdetection/fish2_static",
    "lfimg",
    "gt/LF_Stack",
    "FishStaticLF_488nm_crop1",
    x_roi=(slice(19, -19), slice(19, -39)),
    y_roi=(slice(None), slice(19, -19), slice(19, -39)),
)

fish02_LF2_TP00 = Fish02(
    "20190605_Fish/8214x7943_H2A-mC_H2B-EGFP/fish2_arrythmic/Fish_120to120_241planes/2019-06-05_05.09.18/stack_6_channel_0/TP_00000",
    "RC_rectified",
    "RCout",
    "FishDynamicLF_561nm_TP00",
)

fish02_LF2_TP00_filet = Fish02(
    "20190605_Fish/8214x7943_H2A-mC_H2B-EGFP/fish2_arrythmic/Fish_120to120_241planes/2019-06-05_05.09.18/stack_6_channel_0/TP_00000",
    "RC_rectified",
    "RCout",
    "FishDynamicLF_561nm_TP00",
    y_roi=(slice(7, 42), slice(None), slice(None)),
)

fish02_LF2_TPxx_filets = [Fish02(
    f"20190605_Fish/8214x7943_H2A-mC_H2B-EGFP/fish2_arrythmic/Fish_120to120_241planes/2019-06-05_05.09.18/stack_6_channel_0/TP_{tp:05}",
    "RC_rectified",
    "RCout",
    "FishDynamicLF_561nm_TP00",
    y_roi=(slice(7, 42), slice(None), slice(None)),
) for tp in range(85)]



class RegionalFish(DatasetConfigEntry):
    common_path = Path("/g/kreshuk/beuttenm/Documents/lnet_datasets")
    description = "F02"


regional_fish02_LS1 = RegionalFish(
    "fish02_01",
    "lfimg",
    "light_sheet",
    "FishStaticLS_488nm",
    x_roi=(slice(None), slice(1, None)),
    y_roi=(slice(7, 42), slice(None), slice(1, None)),
)

regional_fish02_RL1 = RegionalFish(
    "fish02_01",
    "lfimg",
    "rl",
    "FishStaticLS_488nm",
    x_roi=(slice(None), slice(1, None)),
    y_roi=(slice(7, 42), slice(None), slice(1, None)),
)
