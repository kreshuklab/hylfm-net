from pathlib import Path

from lnet.config.dataset import NamedDatasetInfo

t0454_TP_00 = NamedDatasetInfo(
    Path("/"),
    "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_04.54.38/stack_1_channel_3/TP_00000/RC_rectified/Cam_Right_*_rectified.tif",
    "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_04.54.38/stack_1_channel_3/TP_00000/LC/Cam_Left_*.tif",
    "t0454",
    # x_shape=(1273, 1463),
    length=241,
    AffineTransform="from_x_path",
    dynamic_z_slice_mod=241,
)
