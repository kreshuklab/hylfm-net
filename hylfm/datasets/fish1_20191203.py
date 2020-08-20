from pathlib import Path

from .base import NamedDatasetInfo, GKRESHUK

t0610_static = NamedDatasetInfo(
    Path(GKRESHUK),
    "LF_partially_restored/LenseLeNet_Microscope/20191203_dynamic_staticHeart_tuesday/fish1/static/Heart_tightCrop/2019-12-03_06.10.28/stack_0_channel_0/TP_*/RC_rectified/Cam_Right_001_rectified.tif",
    "LF_partially_restored/LenseLeNet_Microscope/20191203_dynamic_staticHeart_tuesday/fish1/static/Heart_tightCrop/2019-12-03_06.10.28/stack_1_channel_1/TP_*/LC/Cam_Left_registered.tif",
    "t0610_static",
    length=5,
    x_shape=(1254, 1463),
    y_shape=(838, 1273, 1463),
)
t0610_static_affine = NamedDatasetInfo(
    Path(GKRESHUK),
    "LF_partially_restored/LenseLeNet_Microscope/20191203_dynamic_staticHeart_tuesday/fish1/static/Heart_tightCrop/2019-12-03_06.10.28/stack_0_channel_0/TP_*/RC_rectified/Cam_Right_001_rectified.tif",
    "LF_partially_restored/LenseLeNet_Microscope/20191203_dynamic_staticHeart_tuesday/fish1/static/Heart_tightCrop/2019-12-03_06.10.28/stack_1_channel_1/TP_*/LC/Cam_Left.tif",
    "t0610_affine",
    length=5,
    x_shape=(1254, 1463),
    AffineTransform="from_x_path",
)
