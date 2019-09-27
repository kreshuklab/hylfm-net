from lnet.dataset_configs import DatasetConfigEntry, PathOfInterest


class Platyn00(DatasetConfigEntry):
    description = "platyn_neuro"


platyn1 = Platyn00(
    "",
    "/g/hufnagel/LF/WFdata/Vienna_arendt_collab/20151029/lowOxygen/Rectified/20151029_6dpf_40x_095NA_air_100ms_LED01_pos20_lowOxygen_1",
    "/g/tier2/t2arendt/1_ALUMNI/LukasAnneser/DeconvolvedDatasets/Low_Oxygen/20151029_6dpf_40x_095NA_air_100ms_LED01_pos20_lowOxygen_1_dc",
    "1",
    x_roi=(slice(300, 1290), slice(270, 1560)),
    y_roi=(slice(None), slice(300, 1290), slice(270, 1560)),
    interesting_paths=[
        PathOfInterest((0, 15, 700, 600), (400, 15, 700, 600), (800, 15, 700, 600), (1600, 15, 700, 600)),
        PathOfInterest((0, 15, 800, 1100), (400, 15, 800, 1100), (800, 15, 800, 1100), (1600, 15, 800, 1100)),
    ],
)
platyn2 = Platyn00(
    "",
    "/g/hufnagel/LF/WFdata/Vienna_arendt_collab/20151029/lowOxygen/Rectified/20151029_6dpf_40x_095NA_air_100ms_LED11_pos20_lowOxygen_1",
    "/g/tier2/t2arendt/1_ALUMNI/LukasAnneser/DeconvolvedDatasets/Low_Oxygen/20151029_6dpf_40x_095NA_air_100ms_LED11_pos20_lowOxygen_1_dc",
    "2",
    x_roi=(slice(210, 1290), slice(270, 1545)),
    y_roi=(slice(None), slice(210, 1290), slice(270, 1545)),
    interesting_paths=[
        PathOfInterest((0, 15, 800, 800), (400, 15, 800, 800), (800, 15, 800, 800), (1600, 15, 800, 800))
    ],
)
platyn3 = Platyn00(
    "",
    "/g/hufnagel/LF/WFdata/Vienna_arendt_collab/20151029/lowOxygen/Rectified/20151029_6dpf_40x_095NA_air_100ms_LED15_pos20_lowoxygen_1",
    "/g/tier2/t2arendt/1_ALUMNI/LukasAnneser/DeconvolvedDatasets/Low_Oxygen/20151029_6dpf_40x_095NA_air_100ms_LED15_pos20_lowoxygen_1_dc",
    "3",
    x_roi=(slice(30, 1290), slice(390, 1560)),
    y_roi=(slice(None), slice(30, 1290), slice(390, 1560)),
    interesting_paths=[
        # fake paths
        PathOfInterest((0, 13, 625, 1000), (100, 13, 625, 1000)),
        PathOfInterest((0, 10, 525, 900), (100, 16, 725, 1100)),
        PathOfInterest((0, 0, 700, 1200), (100, 31, 700, 1200)),
        # real paths: (first and last entries are same as next with time distance 30)
        PathOfInterest(
            (250, 13, 625, 1035), (280, 13, 625, 1035), (283, 13, 625, 1030), (286, 14, 622, 1027), (316, 14, 622, 1027)
        ),
        PathOfInterest(
            (250, 13, 584, 1193), (280, 13, 584, 1193), (283, 13, 579, 1192), (286, 14, 622, 1027), (316, 14, 622, 1027)
        ),
        PathOfInterest(
            (170, 9, 698, 1046),
            (200, 9, 698, 1046),
            (260, 9, 697, 1035),
            (320, 8, 698, 1044),
            (340, 8, 698, 1043),
            (370, 8, 698, 1043),
        ),
    ],
)
platyn4 = Platyn00(
    "",
    "/g/hufnagel/LF/WFdata/Vienna_arendt_collab/20151029/lowOxygen/Rectified/20151029_6dpf_40x_095NA_air_100ms_LED23_pos20_lowOxygen_1",
    "/g/tier2/t2arendt/1_ALUMNI/LukasAnneser/DeconvolvedDatasets/Low_Oxygen/20151029_6dpf_40x_095NA_air_100ms_LED23_pos20_lowOxygen_1_dc",
    "4",
    x_roi=(slice(225, 1335), slice(285, 1470)),
    y_roi=(slice(None), slice(225, 1335), slice(285, 1470)),
    interesting_paths=[
        PathOfInterest((0, 15, 800, 800), (400, 15, 800, 800), (800, 15, 800, 800), (1600, 15, 800, 800))
    ],
)

platyn5 = Platyn00(
    "",
    "/g/hufnagel/LF/WFdata/Vienna_arendt_collab/20151029/lowOxygen/Rectified/20151029_6dpf_40x_095NA_air_100ms_LED30_pos20_lowOxygen_1",
    "/g/tier2/t2arendt/1_ALUMNI/LukasAnneser/DeconvolvedDatasets/Low_Oxygen/20151029_6dpf_40x_095NA_air_100ms_LED30_pos20_lowOxygen_1_dc",
    "5",
    x_roi=(slice(75, 1335), slice(315, 1560)),
    y_roi=(slice(None), slice(75, 1335), slice(315, 1560)),
    interesting_paths=[
        PathOfInterest((0, 15, 800, 800), (400, 15, 800, 800), (800, 15, 800, 800), (1600, 15, 800, 800))
    ],
)
