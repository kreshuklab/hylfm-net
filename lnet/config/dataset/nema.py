from pathlib import Path

from lnet.config.dataset import NamedDatasetInfo


nema0_lf_only = NamedDatasetInfo(
    Path("/g/hufnagel/LF/LenseLeNet_Microscope/20191129_tenticles/2019-11-29_03.06.09"),
    "stack_0_channel_0/TP_*/RC_rectified/Cam_Right_1_rectified.tif",
    description="nema0_lf_only",
    length=5
)

nema1_lf_only = NamedDatasetInfo(
    Path("/g/hufnagel/LF/LenseLeNet_Microscope/20191129_tenticles/2019-11-29_03.23.32"),
    "stack_0_channel_0/TP_*/RC_rectified/Cam_Right_1_rectified.tif",
    description="nema1_lf_only",
    length=2
)

nema2_lf_only = NamedDatasetInfo(
    Path("/g/hufnagel/LF/LenseLeNet_Microscope/20191129_tenticles/2019-11-29_03.30.31"),
    "stack_0_channel_0/TP_*/RC_rectified/Cam_Right_1_rectified.tif",
    description="nema2_lf_only",
    length=50
)
