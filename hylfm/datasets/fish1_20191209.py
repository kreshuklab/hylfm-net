from pathlib import Path

from .base import NamedDatasetInfo, GKRESHUK

t0216_static = NamedDatasetInfo(
    Path(GKRESHUK)
    / "LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish1/static/Heart_tightCrop/centered_5steps_stepsize8/2019-12-09_02.16.30",
    "stack_3_channel_0/TP_*/RC_rectified/Cam_Right_001_rectified.tif",
    "stack_4_channel_1/TP_*/LC/Cam_Left_registered.tif",
    "t0216_static",
    length=4,
    x_shape=(1273, 1463),
    y_shape=(838, 1273, 1463),
)
t0216_static_affine = NamedDatasetInfo(
    Path(GKRESHUK)
    / "LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish1/static/Heart_tightCrop/centered_5steps_stepsize8/2019-12-09_02.16.30",
    "stack_3_channel_0/TP_*/RC_rectified/Cam_Right_001_rectified.tif",
    "stack_4_channel_1/TP_*/LC/Cam_Left.tif",
    "t0216_static_affine",
    length=4,
    AffineTransform="from_x_path",
)

t0216_static_lr = NamedDatasetInfo(
    Path(GKRESHUK),
    "Lartially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish1/static/Heart_tightCrop/centered_5steps_stepsize8/2019-12-09_02.16.30/stack_3_channel_0/TP_*/RC_rectified/Cam_Right_001_rectified.tif",
    "LF_computed/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish1/static/Heart_tightCrop/centered_5steps_stepsize8/2019-12-09_02.16.30/stack_3_channel_0/TP_*/RCout/Cam_Right_001.tif",
    "t0216_static_lr",
    length=4,
    x_shape=(1273, 1463),
    y_shape=(838, 1273, 1463),
)


t0223_static = NamedDatasetInfo(
    Path(GKRESHUK)
    / "LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish1/static/Heart_tightCrop/centered_5steps_stepsize8/2019-12-09_02.23.01",
    "stack_3_channel_0/TP_*/RC_rectified/Cam_Right_001_rectified.tif",
    "stack_4_channel_1/TP_*/LC/Cam_Left_registered.tif",
    "t0223_static",
    length=5,
    x_shape=(1273, 1463),
    y_shape=(838, 1273, 1463),
)
t0223_static_affine = NamedDatasetInfo(
    Path(GKRESHUK)
    / "LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish1/static/Heart_tightCrop/centered_5steps_stepsize8/2019-12-09_02.23.01",
    "stack_3_channel_0/TP_*/RC_rectified/Cam_Right_001_rectified.tif",
    "stack_4_channel_1/TP_*/LC/Cam_Left.tif",
    "t0223_static_affine",
    length=5,
    AffineTransform="from_x_path",
)


t0223_static_lr = NamedDatasetInfo(
    Path(GKRESHUK)
    / "Lartially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish1/static/Heart_tightCrop/centered_5steps_stepsize8/2019-12-09_02.23.01/stack_3_channel_0/TP_*/RC_rectified/Cam_Right_001_rectified.tif",
    "LF_computed/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish1/static/Heart_tightCrop/centered_5steps_stepsize8/2019-12-09_02.23.01/stack_3_channel_0/TP_*/RCout/Cam_Right_001.tif",
    "t0223_static_lr",
    length=5,
    x_shape=(1273, 1463),
    y_shape=(838, 1273, 1463),
)

t0229_static = NamedDatasetInfo(
    Path(GKRESHUK)
    / "LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish1/static/Heart_tightCrop/centered_5steps_stepsize8/2019-12-09_02.29.34",
    "stack_3_channel_0/TP_*/RC_rectified/Cam_Right_001_rectified.tif",
    "stack_4_channel_1/TP_*/LC/Cam_Left_registered.tif",
    "t0229_static",
    length=5,
    x_shape=(1273, 1463),
    y_shape=(838, 1273, 1463),
)
t0229_static_affine = NamedDatasetInfo(
    Path(GKRESHUK)
    / "LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish1/static/Heart_tightCrop/centered_5steps_stepsize8/2019-12-09_02.29.34",
    "stack_3_channel_0/TP_*/RC_rectified/Cam_Right_001_rectified.tif",
    "stack_4_channel_1/TP_*/LC/Cam_Left.tif",
    "t0229_static_affine",
    length=5,
    AffineTransform="from_x_path",
)


t0229_static_lr = NamedDatasetInfo(
    Path(GKRESHUK)
    / "Lartially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish1/static/Heart_tightCrop/centered_5steps_stepsize8/2019-12-09_02.29.34/stack_3_channel_0/TP_*/RC_rectified/Cam_Right_001_rectified.tif",
    "LF_computed/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish1/static/Heart_tightCrop/centered_5steps_stepsize8/2019-12-09_02.29.34/stack_3_channel_0/TP_*/RCout/Cam_Right_001.tif",
    "t0229_static_lr",
    length=5,
    x_shape=(1273, 1463),
    y_shape=(838, 1273, 1463),
)

t0235_static = NamedDatasetInfo(
    Path(GKRESHUK)
    / "LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish1/static/Heart_tightCrop/centered_5steps_stepsize8/2019-12-09_02.35.49",
    "stack_3_channel_0/TP_*/RC_rectified/Cam_Right_001_rectified.tif",
    "stack_4_channel_1/TP_*/LC/Cam_Left_registered.tif",
    "t0235_static",
    length=5,
    x_shape=(1273, 1463),
    y_shape=(838, 1273, 1463),
)
t0235_static_affine = NamedDatasetInfo(
    Path(GKRESHUK)
    / "LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish1/static/Heart_tightCrop/centered_5steps_stepsize8/2019-12-09_02.35.49",
    "stack_3_channel_0/TP_*/RC_rectified/Cam_Right_001_rectified.tif",
    "stack_4_channel_1/TP_*/LC/Cam_Left.tif",
    "t0235_static_affine",
    length=5,
    AffineTransform="from_x_path",
)


t0235_static_lr = NamedDatasetInfo(
    Path(GKRESHUK)
    / "Lartially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish1/static/Heart_tightCrop/centered_5steps_stepsize8/2019-12-09_02.35.49/stack_3_channel_0/TP_*/RC_rectified/Cam_Right_001_rectified.tif",
    "LF_computed/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish1/static/Heart_tightCrop/centered_5steps_stepsize8/2019-12-09_02.35.49/stack_3_channel_0/TP_*/RCout/Cam_Right_001.tif",
    "t0235_static_lr",
    length=5,
    x_shape=(1273, 1463),
    y_shape=(838, 1273, 1463),
)

t0242_static = NamedDatasetInfo(
    Path(GKRESHUK)
    / "LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish1/static/Heart_tightCrop/centered_5steps_stepsize8/2019-12-09_02.42.03",
    "stack_3_channel_0/TP_*/RC_rectified/Cam_Right_001_rectified.tif",
    "stack_4_channel_1/TP_*/LC/Cam_Left_registered.tif",
    "t0242_static",
    length=5,
    x_shape=(1273, 1463),
    y_shape=(838, 1273, 1463),
)
t0242_static_affine = NamedDatasetInfo(
    Path(GKRESHUK)
    / "LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish1/static/Heart_tightCrop/centered_5steps_stepsize8/2019-12-09_02.42.03",
    "stack_3_channel_0/TP_*/RC_rectified/Cam_Right_001_rectified.tif",
    "stack_4_channel_1/TP_*/LC/Cam_Left.tif",
    "t0242_static_affine",
    length=5,
    AffineTransform="from_x_path",
)


t0242_static_lr = NamedDatasetInfo(
    Path(GKRESHUK)
    / "Lartially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish1/static/Heart_tightCrop/centered_5steps_stepsize8/2019-12-09_02.42.03/stack_3_channel_0/TP_*/RC_rectified/Cam_Right_001_rectified.tif",
    "LF_computed/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish1/static/Heart_tightCrop/centered_5steps_stepsize8/2019-12-09_02.42.03/stack_3_channel_0/TP_*/RCout/Cam_Right_001.tif",
    "t0242_static_lr",
    length=5,
    x_shape=(1273, 1463),
    y_shape=(838, 1273, 1463),
)

t0248_static = NamedDatasetInfo(
    Path(GKRESHUK)
    / "LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish1/static/Heart_tightCrop/centered_5steps_stepsize8/2019-12-09_02.48.24",
    "stack_3_channel_0/TP_*/RC_rectified/Cam_Right_001_rectified.tif",
    "stack_4_channel_1/TP_*/LC/Cam_Left_registered.tif",
    "t0248_static",
    length=5,
    x_shape=(1273, 1463),
    y_shape=(838, 1273, 1463),
)
t0248_static_affine = NamedDatasetInfo(
    Path(GKRESHUK)
    / "LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish1/static/Heart_tightCrop/centered_5steps_stepsize8/2019-12-09_02.48.24",
    "stack_3_channel_0/TP_*/RC_rectified/Cam_Right_001_rectified.tif",
    "stack_4_channel_1/TP_*/LC/Cam_Left.tif",
    "t0248_static_affine",
    length=5,
    AffineTransform="from_x_path",
)


t0248_static_lr = NamedDatasetInfo(
    Path(GKRESHUK)
    / "Lartially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish1/static/Heart_tightCrop/centered_5steps_stepsize8/2019-12-09_02.48.24/stack_3_channel_0/TP_*/RC_rectified/Cam_Right_001_rectified.tif",
    "LF_computed/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish1/static/Heart_tightCrop/centered_5steps_stepsize8/2019-12-09_02.48.24/stack_3_channel_0/TP_*/RCout/Cam_Right_001.tif",
    "t0248_static_lr",
    length=5,
    x_shape=(1273, 1463),
    y_shape=(838, 1273, 1463),
)

t0254_static = NamedDatasetInfo(
    Path(GKRESHUK)
    / "LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish1/static/Heart_tightCrop/centered_5steps_stepsize8/2019-12-09_02.54.46",
    "stack_3_channel_0/TP_*/RC_rectified/Cam_Right_001_rectified.tif",
    "stack_4_channel_1/TP_*/LC/Cam_Left_registered.tif",
    "fish_20191208_0254_static",
    length=5,
    x_shape=(1273, 1463),
    y_shape=(838, 1273, 1463),
)
t0254_static_affine = NamedDatasetInfo(
    Path(GKRESHUK)
    / "LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish1/static/Heart_tightCrop/centered_5steps_stepsize8/2019-12-09_02.54.46",
    "stack_3_channel_0/TP_*/RC_rectified/Cam_Right_001_rectified.tif",
    "stack_4_channel_1/TP_*/LC/Cam_Left.tif",
    "fish_20191208_0254_static_affine",
    length=5,
    AffineTransform="from_x_path",
)


t0254_static_lr = NamedDatasetInfo(
    Path(GKRESHUK)
    / "Lartially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish1/static/Heart_tightCrop/centered_5steps_stepsize8/2019-12-09_02.54.46/stack_3_channel_0/TP_*/RC_rectified/Cam_Right_001_rectified.tif",
    "LF_computed/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish1/static/Heart_tightCrop/centered_5steps_stepsize8/2019-12-09_02.54.46/stack_3_channel_0/TP_*/RCout/Cam_Right_001.tif",
    "fish_20191208_0254_static_lr",
    length=5,
    x_shape=(1273, 1463),
    y_shape=(838, 1273, 1463),
)
