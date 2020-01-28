from pathlib import Path

from lnet.config.dataset.registration import staticHeartFOV_Transform

from lnet.config.dataset import NamedDatasetInfo

t0815_static = NamedDatasetInfo(
    Path(
        "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/static/Heart_tightCrop/5steps_stepsize8/2019-12-09_08.15.07"
    ),
    "stack_3_channel_0/TP_*/RC_rectified/Cam_Right_001_rectified.tif",
    "stack_4_channel_1/TP_*/LC/Cam_Left_registered.tif",
    "t0815_static",
    length=1,
    x_shape=(1273, 1463),
    y_shape=(838, 1273, 1463),
)
t0815_static_affine = NamedDatasetInfo(
    Path(
        "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/static/Heart_tightCrop/5steps_stepsize8/2019-12-09_08.15.07"
    ),
    "stack_3_channel_0/TP_*/RC_rectified/Cam_Right_001_rectified.tif",
    "stack_4_channel_1/TP_*/LC/Cam_Left.tif",
    "t0815_static_affine",
    length=1,
    AffineTransform="from_x_path",
)
t0815_static_lr = NamedDatasetInfo(
    Path("/g/kreshuk/"),
    "LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/static/Heart_tightCrop/5steps_stepsize8/2019-12-09_08.15.07/stack_3_channel_0/TP_*/RC_rectified/Cam_Right_001_rectified.tif",
    "LF_computed/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/static/Heart_tightCrop/5steps_stepsize8/2019-12-09_08.15.07/stack_3_channel_0/TP_*/RCout/Cam_Right_001.tif",
    "t0815_static_lr",
    length=1,
    x_shape=(1273, 1463),
    y_shape=(838, 1273, 1463),
)
t0819_static = NamedDatasetInfo(
    Path(
        "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/static/Heart_tightCrop/5steps_stepsize8/2019-12-09_08.19.40"
    ),
    "stack_3_channel_0/TP_*/RC_rectified/Cam_Right_001_rectified.tif",
    "stack_4_channel_1/TP_*/LC/Cam_Left_registered.tif",
    "t0819_static",
    length=5,
    x_shape=(1273, 1463),
    y_shape=(838, 1273, 1463),
)
t0819_static_affine = NamedDatasetInfo(
    Path(
        "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/static/Heart_tightCrop/5steps_stepsize8/2019-12-09_08.19.40"
    ),
    "stack_3_channel_0/TP_*/RC_rectified/Cam_Right_001_rectified.tif",
    "stack_4_channel_1/TP_*/LC/Cam_Left.tif",
    "t0819_static_affine",
    length=5,
    AffineTransform="from_x_path",
)

t0819_static_lr = NamedDatasetInfo(
    Path("/g/kreshuk/"),
    "LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/static/Heart_tightCrop/5steps_stepsize8/2019-12-09_08.19.40/stack_3_channel_0/TP_*/RC_rectified/Cam_Right_001_rectified.tif",
    "LF_computed/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/static/Heart_tightCrop/5steps_stepsize8/2019-12-09_08.19.40/stack_3_channel_0/TP_*/RCout/Cam_Right_001.tif",
    "t0819_static_lr",
    length=5,
    x_shape=(1273, 1463),
    y_shape=(838, 1273, 1463),
)

t0827_static = NamedDatasetInfo(
    Path(
        "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/static/Heart_tightCrop/5steps_stepsize8/2019-12-09_08.27.14"
    ),
    "stack_3_channel_0/TP_*/RC_rectified/Cam_Right_001_rectified.tif",
    "stack_4_channel_1/TP_*/LC/Cam_Left_registered.tif",
    "t0827_static",
    length=5,
    x_shape=(1273, 1463),
    y_shape=(838, 1273, 1463),
)
t0827_static_affine = NamedDatasetInfo(
    Path(
        "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/static/Heart_tightCrop/5steps_stepsize8/2019-12-09_08.27.14"
    ),
    "stack_3_channel_0/TP_*/RC_rectified/Cam_Right_001_rectified.tif",
    "stack_4_channel_1/TP_*/LC/Cam_Left.tif",
    "t0827_static_affine",
    length=5,
    AffineTransform="from_x_path",
)

t0827_static_lr = NamedDatasetInfo(
    Path("/g/kreshuk/"),
    "LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/static/Heart_tightCrop/5steps_stepsize8/2019-12-09_08.27.14/stack_3_channel_0/TP_*/RC_rectified/Cam_Right_001_rectified.tif",
    "LF_computed/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/static/Heart_tightCrop/5steps_stepsize8/2019-12-09_08.27.14/stack_3_channel_0/TP_*/RCout/Cam_Right_001.tif",
    "t0827_static_lr",
    length=5,
    x_shape=(1273, 1463),
    y_shape=(838, 1273, 1463),
)

t0834_static = NamedDatasetInfo(
    Path(
        "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/static/Heart_tightCrop/5steps_stepsize15/2019-12-09_08.34.44"
    ),
    "stack_3_channel_0/TP_*/RC_rectified/Cam_Right_001_rectified.tif",
    "stack_4_channel_1/TP_*/LC/Cam_Left_registered.tif",
    "t0834_static",
    length=5,
    x_shape=(1273, 1463),
    y_shape=(838, 1273, 1463),
)
t0834_static_affine = NamedDatasetInfo(
    Path(
        "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/static/Heart_tightCrop/5steps_stepsize15/2019-12-09_08.34.44"
    ),
    "stack_3_channel_0/TP_*/RC_rectified/Cam_Right_001_rectified.tif",
    "stack_4_channel_1/TP_*/LC/Cam_Left.tif",
    "t0834_static_affine",
    length=5,
    AffineTransform="from_x_path",
)

t0834_static_lr = NamedDatasetInfo(
    Path("/g/kreshuk/"),
    "LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/static/Heart_tightCrop/5steps_stepsize15/2019-12-09_08.34.44/stack_3_channel_0/TP_*/RC_rectified/Cam_Right_001_rectified.tif",
    "LF_computed/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/static/Heart_tightCrop/5steps_stepsize15/2019-12-09_08.34.44/stack_3_channel_0/TP_*/RCout/Cam_Right_001.tif",
    "t0834_static_lr",
    length=5,
    x_shape=(1273, 1463),
    y_shape=(838, 1273, 1463),
)

t0841_static = NamedDatasetInfo(
    Path(
        "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/static/Heart_tightCrop/5steps_stepsize15/2019-12-09_08.41.41"
    ),
    "stack_3_channel_0/TP_*/RC_rectified/Cam_Right_001_rectified.tif",
    "stack_4_channel_1/TP_*/LC/Cam_Left_registered.tif",
    "t0841_static",
    length=5,
    x_shape=(1273, 1463),
    y_shape=(838, 1273, 1463),
)
t0841_static_affine = NamedDatasetInfo(
    Path(
        "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/static/Heart_tightCrop/5steps_stepsize15/2019-12-09_08.41.41"
    ),
    "stack_3_channel_0/TP_*/RC_rectified/Cam_Right_001_rectified.tif",
    "stack_4_channel_1/TP_*/LC/Cam_Left.tif",
    "t0841_static_affine",
    length=5,
    AffineTransform="from_x_path",
)

t0841_static_lr = NamedDatasetInfo(
    Path("/g/kreshuk/"),
    "LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/static/Heart_tightCrop/5steps_stepsize15/2019-12-09_08.41.41/stack_3_channel_0/TP_*/RC_rectified/Cam_Right_001_rectified.tif",
    "LF_computed/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/static/Heart_tightCrop/5steps_stepsize15/2019-12-09_08.41.41/stack_3_channel_0/TP_*/RCout/Cam_Right_001.tif",
    "t0841_static_lr",
    length=5,
    x_shape=(1273, 1463),
    y_shape=(838, 1273, 1463),
)

t0851_static = NamedDatasetInfo(
    Path(
        "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/static/Heart_tightCrop/5steps_stepsize15/2019-12-09_08.51.01"
    ),
    "stack_3_channel_0/TP_*/RC_rectified/Cam_Right_001_rectified.tif",
    "stack_4_channel_1/TP_*/LC/Cam_Left_registered.tif",
    "t0851_static",
    length=5,
    x_shape=(1273, 1463),
    y_shape=(838, 1273, 1463),
)
t0851_static_affine = NamedDatasetInfo(
    Path(
        "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/static/Heart_tightCrop/5steps_stepsize15/2019-12-09_08.51.01"
    ),
    "stack_3_channel_0/TP_*/RC_rectified/Cam_Right_001_rectified.tif",
    "stack_4_channel_1/TP_*/LC/Cam_Left.tif",
    "t0851_static_affine",
    length=5,
    AffineTransform="from_x_path",
)

t0851_static_lr = NamedDatasetInfo(
    Path("/g/kreshuk/"),
    "LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/static/Heart_tightCrop/5steps_stepsize15/2019-12-09_08.51.01/stack_3_channel_0/TP_*/RC_rectified/Cam_Right_001_rectified.tif",
    "LF_computed/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/static/Heart_tightCrop/5steps_stepsize15/2019-12-09_08.51.01/stack_3_channel_0/TP_*/RCout/Cam_Right_001.tif",
    "t0851_static_lr",
    length=5,
    x_shape=(1273, 1463),
    y_shape=(838, 1273, 1463),
)

t0901_static = NamedDatasetInfo(
    Path(
        "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/static/Heart_tightCrop/5steps_stepsize15/2019-12-09_09.01.28"
    ),
    "stack_3_channel_0/TP_*/RC_rectified/Cam_Right_001_rectified.tif",
    "stack_4_channel_1/TP_*/LC/Cam_Left_registered.tif",
    "t0901_static",
    length=5,
    x_shape=(1273, 1463),
    y_shape=(838, 1273, 1463),
)
t0901_static_affine = NamedDatasetInfo(
    Path(
        "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/static/Heart_tightCrop/5steps_stepsize15/2019-12-09_09.01.28"
    ),
    "stack_3_channel_0/TP_*/RC_rectified/Cam_Right_001_rectified.tif",
    "stack_4_channel_1/TP_*/LC/Cam_Left.tif",
    "t0901_static_affine",
    length=5,
    AffineTransform="from_x_path",
)

t0901_static_lr = NamedDatasetInfo(
    Path("/g/kreshuk/"),
    "LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/static/Heart_tightCrop/5steps_stepsize15/2019-12-09_09.01.28/stack_3_channel_0/TP_*/RC_rectified/Cam_Right_001_rectified.tif",
    "LF_computed/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/static/Heart_tightCrop/5steps_stepsize15/2019-12-09_09.01.28/stack_3_channel_0/TP_*/RCout/Cam_Right_001.tif",
    "t0901_static_lr",
    length=5,
    x_shape=(1273, 1463),
    y_shape=(838, 1273, 1463),
)

t0911_static = NamedDatasetInfo(
    Path(
        "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/static/Heart_tightCrop/5steps_stepsize15/2019-12-09_09.11.59"
    ),
    "stack_3_channel_0/TP_*/RC_rectified/Cam_Right_001_rectified.tif",
    "stack_4_channel_1/TP_*/LC/Cam_Left_registered.tif",
    "t0911_static",
    length=5,
    x_shape=(1273, 1463),
    y_shape=(838, 1273, 1463),
)
t0911_static_affine = NamedDatasetInfo(
    Path(
        "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/static/Heart_tightCrop/5steps_stepsize15/2019-12-09_09.11.59"
    ),
    "stack_3_channel_0/TP_*/RC_rectified/Cam_Right_001_rectified.tif",
    "stack_4_channel_1/TP_*/LC/Cam_Left.tif",
    "t0911_static_affine",
    length=5,
    AffineTransform="from_x_path",
)

t0911_static_lr = NamedDatasetInfo(
    Path("/g/kreshuk/"),
    "LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/static/Heart_tightCrop/5steps_stepsize15/2019-12-09_09.11.59/stack_3_channel_0/TP_*/RC_rectified/Cam_Right_001_rectified.tif",
    "LF_computed/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/static/Heart_tightCrop/5steps_stepsize15/2019-12-09_09.11.59/stack_3_channel_0/TP_*/RCout/Cam_Right_001.tif",
    "t0911_static_lr",
    length=5,
    x_shape=(1273, 1463),
    y_shape=(838, 1273, 1463),
)

t0918_static = NamedDatasetInfo(
    Path(
        "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/static/Heart_tightCrop/5steps_stepsize15/2019-12-09_09.18.01"
    ),
    "stack_3_channel_0/TP_*/RC_rectified/Cam_Right_001_rectified.tif",
    "stack_4_channel_1/TP_*/LC/Cam_Left_registered.tif",
    "t0918_static",
    length=5,
    x_shape=(1273, 1463),
    y_shape=(838, 1273, 1463),
)
t0918_static_affine = NamedDatasetInfo(
    Path(
        "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/static/Heart_tightCrop/5steps_stepsize15/2019-12-09_09.18.01"
    ),
    "stack_3_channel_0/TP_*/RC_rectified/Cam_Right_001_rectified.tif",
    "stack_4_channel_1/TP_*/LC/Cam_Left.tif",
    "t0918_static_affine",
    length=5,
    AffineTransform="from_x_path",
)

t0918_static_lr = NamedDatasetInfo(
    Path("/g/kreshuk/"),
    "LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/static/Heart_tightCrop/5steps_stepsize15/2019-12-09_09.18.01/stack_3_channel_0/TP_*/RC_rectified/Cam_Right_001_rectified.tif",
    "LF_computed/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/static/Heart_tightCrop/5steps_stepsize15/2019-12-09_09.18.01/stack_3_channel_0/TP_*/RCout/Cam_Right_001.tif",
    "t0918_static_lr",
    length=5,
    x_shape=(1273, 1463),
    y_shape=(838, 1273, 1463),
)
