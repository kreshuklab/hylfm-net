from pathlib import Path

from lnet.config.dataset.registration import Heart_tightCrop_Transform

from lnet.config.dataset import NamedDatasetInfo


fish1_20191208_0216_static = NamedDatasetInfo(
    Path(
        "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish1/static/Heart_tightCrop/centered_5steps_stepsize8/2019-12-09_02.16.30"
    ),
    "stack_3_channel_0/TP_*/RC_rectified/Cam_Right_001_rectified.tif",
    "stack_4_channel_1/TP_*/LC/Cam_Left_registered.tif",
    "fish1_20191208_0216_static",
    length=4,
    y_shape=(838, 1273, 1463),
)
fish1_20191208_0216_static_affine = NamedDatasetInfo(
    Path(
        "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish1/static/Heart_tightCrop/centered_5steps_stepsize8/2019-12-09_02.16.30"
    ),
    "stack_3_channel_0/TP_*/RC_rectified/Cam_Right_001_rectified.tif",
    "stack_4_channel_1/TP_*/LC/Cam_Left.tif",
    "fish1_20191208_0216_static_affine",
    length=4,
    y_shape=(838, 1273, 1463),
    AffineTransform=Heart_tightCrop_Transform,
)

fish1_20191208_0216_static_lr = NamedDatasetInfo(
    Path("/g/kreshuk"),
    "LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish1/static/Heart_tightCrop/centered_5steps_stepsize8/2019-12-09_02.16.30/stack_3_channel_0/TP_*/RC_rectified/Cam_Right_001_rectified.tif",
    "LF_computed/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish1/static/Heart_tightCrop/centered_5steps_stepsize8/2019-12-09_02.16.30/stack_3_channel_0/TP_*/RCout/Cam_Right_001.tif",
    "fish1_20191208_0216_static_lr",
    length=4,
    y_shape=(838, 1273, 1463),
)


fish1_20191208_0223_static = NamedDatasetInfo(
    Path(
        "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish1/static/Heart_tightCrop/centered_5steps_stepsize8/2019-12-09_02.23.01"
    ),
    "stack_3_channel_0/TP_*/RC_rectified/Cam_Right_001_rectified.tif",
    "stack_4_channel_1/TP_*/LC/Cam_Left_registered.tif",
    "fish1_20191208_0223_static",
    length=5,
    y_shape=(838, 1273, 1463),
)
fish1_20191208_0223_static_affine = NamedDatasetInfo(
    Path(
        "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish1/static/Heart_tightCrop/centered_5steps_stepsize8/2019-12-09_02.23.01"
    ),
    "stack_3_channel_0/TP_*/RC_rectified/Cam_Right_001_rectified.tif",
    "stack_4_channel_1/TP_*/LC/Cam_Left.tif",
    "fish1_20191208_0223_static_affine",
    length=5,
    y_shape=(838, 1273, 1463),
    AffineTransform=Heart_tightCrop_Transform,
)


fish1_20191208_0223_static_lr = NamedDatasetInfo(
    Path("/g/kreshuk/"),
    "LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish1/static/Heart_tightCrop/centered_5steps_stepsize8/2019-12-09_02.23.01/stack_3_channel_0/TP_*/RC_rectified/Cam_Right_001_rectified.tif",
    "LF_computed/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish1/static/Heart_tightCrop/centered_5steps_stepsize8/2019-12-09_02.23.01/stack_3_channel_0/TP_*/RCout/Cam_Right_001.tif",
    "fish1_20191208_0223_static_lr",
    length=5,
    y_shape=(838, 1273, 1463),
)

fish1_20191208_0229_static = NamedDatasetInfo(
    Path(
        "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish1/static/Heart_tightCrop/centered_5steps_stepsize8/2019-12-09_02.29.34"
    ),
    "stack_3_channel_0/TP_*/RC_rectified/Cam_Right_001_rectified.tif",
    "stack_4_channel_1/TP_*/LC/Cam_Left_registered.tif",
    "fish1_20191208_0229_static",
    length=5,
    y_shape=(838, 1273, 1463),
)
fish1_20191208_0229_static_affine = NamedDatasetInfo(
    Path(
        "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish1/static/Heart_tightCrop/centered_5steps_stepsize8/2019-12-09_02.29.34"
    ),
    "stack_3_channel_0/TP_*/RC_rectified/Cam_Right_001_rectified.tif",
    "stack_4_channel_1/TP_*/LC/Cam_Left.tif",
    "fish1_20191208_0229_static_affine",
    length=5,
    y_shape=(838, 1273, 1463),
    AffineTransform=Heart_tightCrop_Transform,
)


fish1_20191208_0229_static_lr = NamedDatasetInfo(
    Path("/g/kreshuk/"),
    "LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish1/static/Heart_tightCrop/centered_5steps_stepsize8/2019-12-09_02.29.34/stack_3_channel_0/TP_*/RC_rectified/Cam_Right_001_rectified.tif",
    "LF_computed/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish1/static/Heart_tightCrop/centered_5steps_stepsize8/2019-12-09_02.29.34/stack_3_channel_0/TP_*/RCout/Cam_Right_001.tif",
    "fish1_20191208_0229_static_lr",
    length=5,
    y_shape=(838, 1273, 1463),
)

fish1_20191208_0235_static = NamedDatasetInfo(
    Path(
        "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish1/static/Heart_tightCrop/centered_5steps_stepsize8/2019-12-09_02.35.49"
    ),
    "stack_3_channel_0/TP_*/RC_rectified/Cam_Right_001_rectified.tif",
    "stack_4_channel_1/TP_*/LC/Cam_Left_registered.tif",
    "fish1_20191208_0235_static",
    length=5,
    y_shape=(838, 1273, 1463),
)
fish1_20191208_0235_static_affine = NamedDatasetInfo(
    Path(
        "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish1/static/Heart_tightCrop/centered_5steps_stepsize8/2019-12-09_02.35.49"
    ),
    "stack_3_channel_0/TP_*/RC_rectified/Cam_Right_001_rectified.tif",
    "stack_4_channel_1/TP_*/LC/Cam_Left.tif",
    "fish1_20191208_0235_static_affine",
    length=5,
    y_shape=(838, 1273, 1463),
    AffineTransform=Heart_tightCrop_Transform,
)


fish1_20191208_0235_static_lr = NamedDatasetInfo(
    Path("/g/kreshuk/"),
    "LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish1/static/Heart_tightCrop/centered_5steps_stepsize8/2019-12-09_02.35.49/stack_3_channel_0/TP_*/RC_rectified/Cam_Right_001_rectified.tif",
    "LF_computed/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish1/static/Heart_tightCrop/centered_5steps_stepsize8/2019-12-09_02.35.49/stack_3_channel_0/TP_*/RCout/Cam_Right_001.tif",
    "fish1_20191208_0235_static_lr",
    length=5,
    y_shape=(838, 1273, 1463),
)

fish1_20191208_0242_static = NamedDatasetInfo(
    Path(
        "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish1/static/Heart_tightCrop/centered_5steps_stepsize8/2019-12-09_02.42.03"
    ),
    "stack_3_channel_0/TP_*/RC_rectified/Cam_Right_001_rectified.tif",
    "stack_4_channel_1/TP_*/LC/Cam_Left_registered.tif",
    "fish1_20191208_0242_static",
    length=5,
    y_shape=(838, 1273, 1463),
)
fish1_20191208_0242_static_affine = NamedDatasetInfo(
    Path(
        "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish1/static/Heart_tightCrop/centered_5steps_stepsize8/2019-12-09_02.42.03"
    ),
    "stack_3_channel_0/TP_*/RC_rectified/Cam_Right_001_rectified.tif",
    "stack_4_channel_1/TP_*/LC/Cam_Left.tif",
    "fish1_20191208_0242_static_affine",
    length=5,
    y_shape=(838, 1273, 1463),
    AffineTransform=Heart_tightCrop_Transform,
)


fish1_20191208_0242_static_lr = NamedDatasetInfo(
    Path("/g/kreshuk/"),
    "LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish1/static/Heart_tightCrop/centered_5steps_stepsize8/2019-12-09_02.42.03/stack_3_channel_0/TP_*/RC_rectified/Cam_Right_001_rectified.tif",
    "LF_computed/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish1/static/Heart_tightCrop/centered_5steps_stepsize8/2019-12-09_02.42.03/stack_3_channel_0/TP_*/RCout/Cam_Right_001.tif",
    "fish1_20191208_0242_static_lr",
    length=5,
    y_shape=(838, 1273, 1463),
)

fish1_20191208_0248_static = NamedDatasetInfo(
    Path(
        "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish1/static/Heart_tightCrop/centered_5steps_stepsize8/2019-12-09_02.48.24"
    ),
    "stack_3_channel_0/TP_*/RC_rectified/Cam_Right_001_rectified.tif",
    "stack_4_channel_1/TP_*/LC/Cam_Left_registered.tif",
    "fish1_20191208_0248_static",
    length=5,
    y_shape=(838, 1273, 1463),
)
fish1_20191208_0248_static_affine = NamedDatasetInfo(
    Path(
        "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish1/static/Heart_tightCrop/centered_5steps_stepsize8/2019-12-09_02.48.24"
    ),
    "stack_3_channel_0/TP_*/RC_rectified/Cam_Right_001_rectified.tif",
    "stack_4_channel_1/TP_*/LC/Cam_Left.tif",
    "fish1_20191208_0248_static_affine",
    length=5,
    y_shape=(838, 1273, 1463),
    AffineTransform=Heart_tightCrop_Transform,
)


fish1_20191208_0248_static_lr = NamedDatasetInfo(
    Path("/g/kreshuk/"),
    "LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish1/static/Heart_tightCrop/centered_5steps_stepsize8/2019-12-09_02.48.24/stack_3_channel_0/TP_*/RC_rectified/Cam_Right_001_rectified.tif",
    "LF_computed/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish1/static/Heart_tightCrop/centered_5steps_stepsize8/2019-12-09_02.48.24/stack_3_channel_0/TP_*/RCout/Cam_Right_001.tif",
    "fish1_20191208_0248_static_lr",
    length=5,
    y_shape=(838, 1273, 1463),
)

fish1_20191208_0254_static = NamedDatasetInfo(
    Path(
        "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish1/static/Heart_tightCrop/centered_5steps_stepsize8/2019-12-09_02.54.46"
    ),
    "stack_3_channel_0/TP_*/RC_rectified/Cam_Right_001_rectified.tif",
    "stack_4_channel_1/TP_*/LC/Cam_Left_registered.tif",
    "fish_20191208_0254_static",
    length=5,
    y_shape=(838, 1273, 1463),
)
fish1_20191208_0254_static_affine = NamedDatasetInfo(
    Path(
        "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish1/static/Heart_tightCrop/centered_5steps_stepsize8/2019-12-09_02.54.46"
    ),
    "stack_3_channel_0/TP_*/RC_rectified/Cam_Right_001_rectified.tif",
    "stack_4_channel_1/TP_*/LC/Cam_Left.tif",
    "fish_20191208_0254_static_affine",
    length=5,
    y_shape=(838, 1273, 1463),
    AffineTransform=Heart_tightCrop_Transform,
)


fish1_20191208_0254_static_lr = NamedDatasetInfo(
    Path("/g/kreshuk/"),
    "LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish1/static/Heart_tightCrop/centered_5steps_stepsize8/2019-12-09_02.54.46/stack_3_channel_0/TP_*/RC_rectified/Cam_Right_001_rectified.tif",
    "LF_computed/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish1/static/Heart_tightCrop/centered_5steps_stepsize8/2019-12-09_02.54.46/stack_3_channel_0/TP_*/RCout/Cam_Right_001.tif",
    "fish_20191208_0254_static_lr",
    length=5,
    y_shape=(838, 1273, 1463),
)

# fish2
fish2_20191208_0815_static = NamedDatasetInfo(
    Path(
        "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/static/Heart_tightCrop/5steps_stepsize8/2019-12-09_08.15.07"
    ),
    "stack_3_channel_0/TP_*/RC_rectified/Cam_Right_001_rectified.tif",
    "stack_4_channel_1/TP_*/LC/Cam_Left_registered.tif",
    "fish2_20191208_0815_static",
    length=1,
    x_shape=(1273, 1463),
    y_shape=(838, 1273, 1463),
)
fish2_20191208_0815_static_affine = NamedDatasetInfo(
    Path(
        "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/static/Heart_tightCrop/5steps_stepsize8/2019-12-09_08.15.07"
    ),
    "stack_3_channel_0/TP_*/RC_rectified/Cam_Right_001_rectified.tif",
    "stack_4_channel_1/TP_*/LC/Cam_Left.tif",
    "fish2_20191208_0815_static_affine",
    length=1,
    x_shape=(1273, 1463),
    y_shape=(838, 1273, 1463),
    AffineTransform=Heart_tightCrop_Transform,
)
fish2_20191208_0815_static_lr = NamedDatasetInfo(
    Path("/g/kreshuk/"),
    "LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/static/Heart_tightCrop/5steps_stepsize8/2019-12-09_08.15.07/stack_3_channel_0/TP_*/RC_rectified/Cam_Right_001_rectified.tif",
    "LF_computed/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/static/Heart_tightCrop/5steps_stepsize8/2019-12-09_08.15.07/stack_3_channel_0/TP_*/RCout/Cam_Right_001.tif",
    "fish2_20191208_0815_static_lr",
    length=1,
    x_shape=(1273, 1463),
    y_shape=(838, 1273, 1463),
)

fish2_20191209_0819_static = NamedDatasetInfo(
    Path(
        "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/static/Heart_tightCrop/5steps_stepsize8/2019-12-09_08.19.40"
    ),
    "stack_3_channel_0/TP_*/RC_rectified/Cam_Right_001_rectified.tif",
    "stack_4_channel_1/TP_*/LC/Cam_Left_registered.tif",
    "fish2_20191209_0819_static",
    length=5,
    x_shape=(1273, 1463),
    y_shape=(838, 1273, 1463),
)
fish2_20191209_0819_static_affine = NamedDatasetInfo(
    Path(
        "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/static/Heart_tightCrop/5steps_stepsize8/2019-12-09_08.19.40"
    ),
    "stack_3_channel_0/TP_*/RC_rectified/Cam_Right_001_rectified.tif",
    "stack_4_channel_1/TP_*/LC/Cam_Left.tif",
    "fish2_20191209_0819_static_affine",
    length=5,
    x_shape=(1273, 1463),
    y_shape=(838, 1273, 1463),
    AffineTransform=Heart_tightCrop_Transform,
)

fish2_20191209_0819_static_lr = NamedDatasetInfo(
    Path("/g/kreshuk/"),
    "LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/static/Heart_tightCrop/5steps_stepsize8/2019-12-09_08.19.40/stack_3_channel_0/TP_*/RC_rectified/Cam_Right_001_rectified.tif",
    "LF_computed/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/static/Heart_tightCrop/5steps_stepsize8/2019-12-09_08.19.40/stack_3_channel_0/TP_*/RCout/Cam_Right_001.tif",
    "fish2_20191209_0819_static_lr",
    length=5,
    x_shape=(1273, 1463),
    y_shape=(838, 1273, 1463),
)

fish2_20191209_0827_static = NamedDatasetInfo(
    Path(
        "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/static/Heart_tightCrop/5steps_stepsize8/2019-12-09_08.27.14"
    ),
    "stack_3_channel_0/TP_*/RC_rectified/Cam_Right_001_rectified.tif",
    "stack_4_channel_1/TP_*/LC/Cam_Left_registered.tif",
    "fish2_20191209_0827_static",
    length=5,
    x_shape=(1273, 1463),
    y_shape=(838, 1273, 1463),
)
fish2_20191209_0827_static_affine = NamedDatasetInfo(
    Path(
        "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/static/Heart_tightCrop/5steps_stepsize8/2019-12-09_08.27.14"
    ),
    "stack_3_channel_0/TP_*/RC_rectified/Cam_Right_001_rectified.tif",
    "stack_4_channel_1/TP_*/LC/Cam_Left.tif",
    "fish2_20191209_0827_static_affine",
    length=5,
    x_shape=(1273, 1463),
    y_shape=(838, 1273, 1463),
    AffineTransform=Heart_tightCrop_Transform,
)

fish2_20191209_0827_static_lr = NamedDatasetInfo(
    Path("/g/kreshuk/"),
    "LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/static/Heart_tightCrop/5steps_stepsize8/2019-12-09_08.27.14/stack_3_channel_0/TP_*/RC_rectified/Cam_Right_001_rectified.tif",
    "LF_computed/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/static/Heart_tightCrop/5steps_stepsize8/2019-12-09_08.27.14/stack_3_channel_0/TP_*/RCout/Cam_Right_001.tif",
    "fish2_20191209_0827_static_lr",
    length=5,
    x_shape=(1273, 1463),
    y_shape=(838, 1273, 1463),
)

fish2_20191209_0834_static = NamedDatasetInfo(
    Path(
        "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/static/Heart_tightCrop/5steps_stepsize15/2019-12-09_08.34.44"
    ),
    "stack_3_channel_0/TP_*/RC_rectified/Cam_Right_001_rectified.tif",
    "stack_4_channel_1/TP_*/LC/Cam_Left_registered.tif",
    "fish2_20191209_0834_static",
    length=5,
    x_shape=(1273, 1463),
    y_shape=(838, 1273, 1463),
)
fish2_20191209_0834_static_affine = NamedDatasetInfo(
    Path(
        "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/static/Heart_tightCrop/5steps_stepsize15/2019-12-09_08.34.44"
    ),
    "stack_3_channel_0/TP_*/RC_rectified/Cam_Right_001_rectified.tif",
    "stack_4_channel_1/TP_*/LC/Cam_Left.tif",
    "fish2_20191209_0834_static_affine",
    length=5,
    x_shape=(1273, 1463),
    y_shape=(838, 1273, 1463),
    AffineTransform=Heart_tightCrop_Transform,
)

fish2_20191209_0834_static_lr = NamedDatasetInfo(
    Path("/g/kreshuk/"),
    "LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/static/Heart_tightCrop/5steps_stepsize15/2019-12-09_08.34.44/stack_3_channel_0/TP_*/RC_rectified/Cam_Right_001_rectified.tif",
    "LF_computed/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/static/Heart_tightCrop/5steps_stepsize15/2019-12-09_08.34.44/stack_3_channel_0/TP_*/RCout/Cam_Right_001.tif",
    "fish2_20191209_0834_static_lr",
    length=5,
    x_shape=(1273, 1463),
    y_shape=(838, 1273, 1463),
)

fish2_20191209_0841_static = NamedDatasetInfo(
    Path(
        "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/static/Heart_tightCrop/5steps_stepsize15/2019-12-09_08.41.41"
    ),
    "stack_3_channel_0/TP_*/RC_rectified/Cam_Right_001_rectified.tif",
    "stack_4_channel_1/TP_*/LC/Cam_Left_registered.tif",
    "fish2_20191209_0841_static",
    length=5,
    x_shape=(1273, 1463),
    y_shape=(838, 1273, 1463),
)
fish2_20191209_0841_static_affine = NamedDatasetInfo(
    Path(
        "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/static/Heart_tightCrop/5steps_stepsize15/2019-12-09_08.41.41"
    ),
    "stack_3_channel_0/TP_*/RC_rectified/Cam_Right_001_rectified.tif",
    "stack_4_channel_1/TP_*/LC/Cam_Left.tif",
    "fish2_20191209_0841_static_affine",
    length=5,
    x_shape=(1273, 1463),
    y_shape=(838, 1273, 1463),
    AffineTransform=Heart_tightCrop_Transform,
)

fish2_20191209_0841_static_lr = NamedDatasetInfo(
    Path("/g/kreshuk/"),
    "LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/static/Heart_tightCrop/5steps_stepsize15/2019-12-09_08.41.41/stack_3_channel_0/TP_*/RC_rectified/Cam_Right_001_rectified.tif",
    "LF_computed/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/static/Heart_tightCrop/5steps_stepsize15/2019-12-09_08.41.41/stack_3_channel_0/TP_*/RCout/Cam_Right_001.tif",
    "fish2_20191209_0841_static_lr",
    length=5,
    x_shape=(1273, 1463),
    y_shape=(838, 1273, 1463),
)

fish2_20191209_0851_static = NamedDatasetInfo(
    Path(
        "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/static/Heart_tightCrop/5steps_stepsize15/2019-12-09_08.51.01"
    ),
    "stack_3_channel_0/TP_*/RC_rectified/Cam_Right_001_rectified.tif",
    "stack_4_channel_1/TP_*/LC/Cam_Left_registered.tif",
    "fish2_20191209_0851_static",
    length=5,
    x_shape=(1273, 1463),
    y_shape=(838, 1273, 1463),
)
fish2_20191209_0851_static_affine = NamedDatasetInfo(
    Path(
        "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/static/Heart_tightCrop/5steps_stepsize15/2019-12-09_08.51.01"
    ),
    "stack_3_channel_0/TP_*/RC_rectified/Cam_Right_001_rectified.tif",
    "stack_4_channel_1/TP_*/LC/Cam_Left.tif",
    "fish2_20191209_0851_static_affine",
    length=5,
    x_shape=(1273, 1463),
    y_shape=(838, 1273, 1463),
    AffineTransform=Heart_tightCrop_Transform,
)

fish2_20191209_0851_static_lr = NamedDatasetInfo(
    Path("/g/kreshuk/"),
    "LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/static/Heart_tightCrop/5steps_stepsize15/2019-12-09_08.51.01/stack_3_channel_0/TP_*/RC_rectified/Cam_Right_001_rectified.tif",
    "LF_computed/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/static/Heart_tightCrop/5steps_stepsize15/2019-12-09_08.51.01/stack_3_channel_0/TP_*/RCout/Cam_Right_001.tif",
    "fish2_20191209_0851_static_lr",
    length=5,
    x_shape=(1273, 1463),
    y_shape=(838, 1273, 1463),
)

fish2_20191209_0901_static = NamedDatasetInfo(
    Path(
        "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/static/Heart_tightCrop/5steps_stepsize15/2019-12-09_09.01.28"
    ),
    "stack_3_channel_0/TP_*/RC_rectified/Cam_Right_001_rectified.tif",
    "stack_4_channel_1/TP_*/LC/Cam_Left_registered.tif",
    "fish2_20191209_0901_static",
    length=5,
    x_shape=(1273, 1463),
    y_shape=(838, 1273, 1463),
)
fish2_20191209_0901_static_affine = NamedDatasetInfo(
    Path(
        "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/static/Heart_tightCrop/5steps_stepsize15/2019-12-09_09.01.28"
    ),
    "stack_3_channel_0/TP_*/RC_rectified/Cam_Right_001_rectified.tif",
    "stack_4_channel_1/TP_*/LC/Cam_Left.tif",
    "fish2_20191209_0901_static_affine",
    length=5,
    x_shape=(1273, 1463),
    y_shape=(838, 1273, 1463),
    AffineTransform=Heart_tightCrop_Transform,
)

fish2_20191209_0901_static_lr = NamedDatasetInfo(
    Path("/g/kreshuk/"),
    "LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/static/Heart_tightCrop/5steps_stepsize15/2019-12-09_09.01.28/stack_3_channel_0/TP_*/RC_rectified/Cam_Right_001_rectified.tif",
    "LF_computed/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/static/Heart_tightCrop/5steps_stepsize15/2019-12-09_09.01.28/stack_3_channel_0/TP_*/RCout/Cam_Right_001.tif",
    "fish2_20191209_0901_static_lr",
    length=5,
    x_shape=(1273, 1463),
    y_shape=(838, 1273, 1463),
)

fish2_20191209_0911_static = NamedDatasetInfo(
    Path(
        "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/static/Heart_tightCrop/5steps_stepsize15/2019-12-09_09.11.59"
    ),
    "stack_3_channel_0/TP_*/RC_rectified/Cam_Right_001_rectified.tif",
    "stack_4_channel_1/TP_*/LC/Cam_Left_registered.tif",
    "fish2_20191209_0911_static",
    length=5,
    x_shape=(1273, 1463),
    y_shape=(838, 1273, 1463),
)
fish2_20191209_0911_static_affine = NamedDatasetInfo(
    Path(
        "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/static/Heart_tightCrop/5steps_stepsize15/2019-12-09_09.11.59"
    ),
    "stack_3_channel_0/TP_*/RC_rectified/Cam_Right_001_rectified.tif",
    "stack_4_channel_1/TP_*/LC/Cam_Left.tif",
    "fish2_20191209_0911_static_affine",
    length=5,
    x_shape=(1273, 1463),
    y_shape=(838, 1273, 1463),
    AffineTransform=Heart_tightCrop_Transform,
)

fish2_20191209_0911_static_lr = NamedDatasetInfo(
    Path("/g/kreshuk/"),
    "LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/static/Heart_tightCrop/5steps_stepsize15/2019-12-09_09.11.59/stack_3_channel_0/TP_*/RC_rectified/Cam_Right_001_rectified.tif",
    "LF_computed/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/static/Heart_tightCrop/5steps_stepsize15/2019-12-09_09.11.59/stack_3_channel_0/TP_*/RCout/Cam_Right_001.tif",
    "fish2_20191209_0911_static_lr",
    length=5,
    x_shape=(1273, 1463),
    y_shape=(838, 1273, 1463),
)

fish2_20191209_0918_static = NamedDatasetInfo(
    Path(
        "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/static/Heart_tightCrop/5steps_stepsize15/2019-12-09_09.18.01"
    ),
    "stack_3_channel_0/TP_*/RC_rectified/Cam_Right_001_rectified.tif",
    "stack_4_channel_1/TP_*/LC/Cam_Left_registered.tif",
    "fish2_20191209_0918_static",
    length=5,
    x_shape=(1273, 1463),
    y_shape=(838, 1273, 1463),
)
fish2_20191209_0918_static_affine = NamedDatasetInfo(
    Path(
        "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/static/Heart_tightCrop/5steps_stepsize15/2019-12-09_09.18.01"
    ),
    "stack_3_channel_0/TP_*/RC_rectified/Cam_Right_001_rectified.tif",
    "stack_4_channel_1/TP_*/LC/Cam_Left.tif",
    "fish2_20191209_0918_static_affine",
    length=5,
    x_shape=(1273, 1463),
    y_shape=(838, 1273, 1463),
    AffineTransform=Heart_tightCrop_Transform,
)

fish2_20191209_0918_static_lr = NamedDatasetInfo(
    Path("/g/kreshuk/"),
    "LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/static/Heart_tightCrop/5steps_stepsize15/2019-12-09_09.18.01/stack_3_channel_0/TP_*/RC_rectified/Cam_Right_001_rectified.tif",
    "LF_computed/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/static/Heart_tightCrop/5steps_stepsize15/2019-12-09_09.18.01/stack_3_channel_0/TP_*/RCout/Cam_Right_001.tif",
    "fish2_20191209_0918_static_lr",
    length=5,
    x_shape=(1273, 1463),
    y_shape=(838, 1273, 1463),
)

# fish3
fish3_20191210_0424_static = NamedDatasetInfo(
    Path("/g/kreshuk"),
    "LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish3/static/staticHeartFOV/20steps_stepsize5/2019-12-10_04.24.29/stack_3_channel_0/TP_*/RC_rectified/Cam_Right_001_rectified.tif",
    "LF_computed/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish3/static/staticHeartFOV/20steps_stepsize5/2019-12-10_04.24.29/stack_4_channel_1/TP_*/LC/Cam_Left_registered.tif",
    "fish3_20191210_0424_static",
    length=20,
    x_shape=(1178, 1767),
    y_shape=(838, 1178, 1767),
)

fish3_20191210_0424_static_lr = NamedDatasetInfo(
    Path("/g/kreshuk"),
    "LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish3/static/staticHeartFOV/20steps_stepsize5/2019-12-10_04.24.29/stack_3_channel_0/TP_*/RC_rectified/Cam_Right_001_rectified.tif",
    "LF_computed/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish3/static/staticHeartFOV/20steps_stepsize5/2019-12-10_04.24.29/stack_3_channel_0/TP_*/RCout/Cam_Right_001.tif",
    "fish3_20191210_0424_static_lr",
    length=20,
    x_shape=(1178, 1767),
    y_shape=(838, 1178, 1767),
)

fish3_20191210_0514_static = NamedDatasetInfo(
    Path("/g/kreshuk"),
    "LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish3/static/staticHeartFOV/20steps_stepsize5/2019-12-10_05.14.57/stack_3_channel_0/TP_*/RC_rectified/Cam_Right_001_rectified.tif",
    "LF_computed/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish3/static/staticHeartFOV/20steps_stepsize5/2019-12-10_05.14.57/stack_4_channel_1/TP_*/LC/Cam_Left_registered.tif",
    "fish3_20191210_0514_static",
    length=20,
    x_shape=(1178, 1767),
    y_shape=(838, 1178, 1767),
)

fish3_20191210_0514_static_lr = NamedDatasetInfo(
    Path("/g/kreshuk"),
    "LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish3/static/staticHeartFOV/20steps_stepsize5/2019-12-10_05.14.57/stack_3_channel_0/TP_*/RC_rectified/Cam_Right_001_rectified.tif",
    "LF_computed/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish3/static/staticHeartFOV/20steps_stepsize5/2019-12-10_05.14.57/stack_3_channel_0/TP_*/RCout/Cam_Right_001.tif",
    "fish3_20191210_0514_static_lr",
    length=20,
    x_shape=(1178, 1767),
    y_shape=(838, 1178, 1767),
)

fish3_20191210_0541_static = NamedDatasetInfo(
    Path("/g/kreshuk"),
    "LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish3/static/staticHeartFOV/20steps_stepsize5/2019-12-10_05.41.48/stack_3_channel_0/TP_*/RC_rectified/Cam_Right_001_rectified.tif",
    "LF_computed/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish3/static/staticHeartFOV/20steps_stepsize5/2019-12-10_05.41.48/stack_4_channel_1/TP_*/LC/Cam_Left_registered.tif",
    "fish3_20191210_0541_static",
    length=20,
    x_shape=(1178, 1767),
    y_shape=(838, 1178, 1767),
)

fish3_20191210_0541_static_lr = NamedDatasetInfo(
    Path("/g/kreshuk"),
    "LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish3/static/staticHeartFOV/20steps_stepsize5/2019-12-10_05.41.48/stack_3_channel_0/TP_*/RC_rectified/Cam_Right_001_rectified.tif",
    "LF_computed/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish3/static/staticHeartFOV/20steps_stepsize5/2019-12-10_05.41.48/stack_3_channel_0/TP_*/RCout/Cam_Right_001.tif",
    "fish3_20191210_0541_static_lr",
    length=20,
    x_shape=(1178, 1767),
    y_shape=(838, 1178, 1767),
)

fish3_20191210_0603_static = NamedDatasetInfo(
    Path("/g/kreshuk"),
    "LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish3/static/staticHeartFOV/20steps_stepsize5/2019-12-10_06.03.37/stack_3_channel_0/TP_*/RC_rectified/Cam_Right_001_rectified.tif",
    "LF_computed/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish3/static/staticHeartFOV/20steps_stepsize5/2019-12-10_06.03.37/stack_4_channel_1/TP_*/LC/Cam_Left_registered.tif",
    "fish3_20191210_0603_static",
    length=20,
    x_shape=(1178, 1767),
    y_shape=(838, 1178, 1767),
)

fish3_20191210_0603_static_lr = NamedDatasetInfo(
    Path("/g/kreshuk"),
    "LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish3/static/staticHeartFOV/20steps_stepsize5/2019-12-10_06.03.37/stack_3_channel_0/TP_*/RC_rectified/Cam_Right_001_rectified.tif",
    "LF_computed/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish3/static/staticHeartFOV/20steps_stepsize5/2019-12-10_06.03.37/stack_3_channel_0/TP_*/RCout/Cam_Right_001.tif",
    "fish3_20191210_0603_static_lr",
    length=20,
    x_shape=(1178, 1767),
    y_shape=(838, 1178, 1767),
)

fish3_20191210_0625_static = NamedDatasetInfo(
    Path("/g/kreshuk"),
    "LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish3/static/staticHeartFOV/20steps_stepsize5/2019-12-10_06.25.14/stack_3_channel_0/TP_*/RC_rectified/Cam_Right_001_rectified.tif",
    "LF_computed/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish3/static/staticHeartFOV/20steps_stepsize5/2019-12-10_06.25.14/stack_4_channel_1/TP_*/LC/Cam_Left_registered.tif",
    "fish3_20191210_0625_static",
    length=20,
    x_shape=(1178, 1767),
    y_shape=(838, 1178, 1767),
)

fish3_20191210_0625_static_lr = NamedDatasetInfo(
    Path("/g/kreshuk"),
    "LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish3/static/staticHeartFOV/20steps_stepsize5/2019-12-10_06.25.14/stack_3_channel_0/TP_*/RC_rectified/Cam_Right_001_rectified.tif",
    "LF_computed/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish3/static/staticHeartFOV/20steps_stepsize5/2019-12-10_06.25.14/stack_3_channel_0/TP_*/RCout/Cam_Right_001.tif",
    "fish3_20191210_0625_static_lr",
    length=20,
    x_shape=(1178, 1767),
    y_shape=(838, 1178, 1767),
)

#
# for i in range(20):
#     print(f'ls -l "/g/kreshuk/LF_computed/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish3/static/staticHeartFOV/20steps_stepsize5/2019-12-10_06.25.14/stack_4_channel_1/TP_000{i:02}/LC/Cam_Left_registered.tif"')

# for i in range(5):
#     print(f"ls -l /g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/static/Heart_tightCrop/5steps_stepsize15/2019-12-09_08.34.44/stack_4_channel_1/TP_{i:05}/LC/Cam_Left_registered.tif")
#     print(f"ls -l /g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/static/Heart_tightCrop/5steps_stepsize15/2019-12-09_08.41.41/stack_4_channel_1/TP_{i:05}/LC/Cam_Left_registered.tif")
#     print(f"ls -l /g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/static/Heart_tightCrop/5steps_stepsize15/2019-12-09_08.51.01/stack_4_channel_1/TP_{i:05}/LC/Cam_Left_registered.tif")
#     print(f"ls -l /g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/static/Heart_tightCrop/5steps_stepsize15/2019-12-09_09.01.28/stack_4_channel_1/TP_{i:05}/LC/Cam_Left_registered.tif")
#     print(f"ls -l /g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/static/Heart_tightCrop/5steps_stepsize15/2019-12-09_09.11.59/stack_4_channel_1/TP_{i:05}/LC/Cam_Left_registered.tif")
