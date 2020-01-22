from pathlib import Path

from lnet.config.dataset.registration import wholeFOV_Transform

from lnet.config.dataset import NamedDatasetInfo

fish1_20191209_0216_static = NamedDatasetInfo(
    Path(
        "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish1/static/wholeFOV/centered_5steps_stepsize8/2019-12-09_02.16.30"
    ),
    "stack_3_channel_0/TP_*/RC_rectified/Cam_Right_001_rectified.tif",
    "stack_4_channel_1/TP_*/LC/Cam_Left.tif",
    "fish1_20191209_0216_static",
)
fish1_20191209_0216_static_affine = NamedDatasetInfo(
    Path(
        "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish1/static/wholeFOV/centered_5steps_stepsize8/2019-12-09_02.16.30"
    ),
    "stack_3_channel_0/TP_*/RC_rectified/Cam_Right_001_rectified.tif",
    "stack_4_channel_1/TP_*/LC/Cam_Left.tif",
    "fish1_20191209_0216_static_affine",
    AffineTransform=wholeFOV_Transform,
)
fish1_20191209_0216_static_lr = NamedDatasetInfo(
    Path("/g/kreshuk"),
    "LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish1/static/wholeFOV/centered_5steps_stepsize8/2019-12-09_02.16.30/stack_3_channel_0/TP_*/RC_rectified/Cam_Right_001_rectified.tif",
    "LF_computed/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish1/static/wholeFOV/centered_5steps_stepsize8/2019-12-09_02.16.30/stack_3_channel_0/TP_*/RCout/Cam_Right_001.tif",
    "fish1_20191209_0216_static_lr",
)

fish1_20191209_0223_static = NamedDatasetInfo(
    Path(
        "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish1/static/wholeFOV/centered_5steps_stepsize8/2019-12-09_02.23.01"
    ),
    "stack_3_channel_0/TP_*/RC_rectified/Cam_Right_001_rectified.tif",
    "stack_4_channel_1/TP_*/LC/Cam_Left.tif",
    "fish1_20191209_0223_static",
)
fish1_20191209_0223_static_affine = NamedDatasetInfo(
    Path(
        "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish1/static/wholeFOV/centered_5steps_stepsize8/2019-12-09_02.23.01"
    ),
    "stack_3_channel_0/TP_*/RC_rectified/Cam_Right_001_rectified.tif",
    "stack_4_channel_1/TP_*/LC/Cam_Left.tif",
    "fish1_20191209_0223_static_affine",
    AffineTransform=wholeFOV_Transform,
)
fish1_20191209_0223_static_lr = NamedDatasetInfo(
    Path("/g/kreshuk"),
    "LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish1/static/wholeFOV/centered_5steps_stepsize8/2019-12-09_02.23.01/stack_3_channel_0/TP_*/RC_rectified/Cam_Right_001_rectified.tif",
    "LF_computed/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish1/static/wholeFOV/centered_5steps_stepsize8/2019-12-09_02.23.01/stack_3_channel_0/TP_*/RCout/Cam_Right_001.tif",
    "fish1_20191209_0223_static_lr",
)

fish1_20191209_0229_static = NamedDatasetInfo(
    Path(
        "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish1/static/wholeFOV/centered_5steps_stepsize8/2019-12-09_02.29.34"
    ),
    "stack_3_channel_0/TP_*/RC_rectified/Cam_Right_001_rectified.tif",
    "stack_4_channel_1/TP_*/LC/Cam_Left.tif",
    "fish1_20191209_0229_static",
)
fish1_20191209_0229_static_affine = NamedDatasetInfo(
    Path(
        "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish1/static/wholeFOV/centered_5steps_stepsize8/2019-12-09_02.29.34"
    ),
    "stack_3_channel_0/TP_*/RC_rectified/Cam_Right_001_rectified.tif",
    "stack_4_channel_1/TP_*/LC/Cam_Left.tif",
    "fish1_20191209_0229_static_affine",
    AffineTransform=wholeFOV_Transform,
)
fish1_20191209_0229_static_lr = NamedDatasetInfo(
    Path("/g/kreshuk"),
    "LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish1/static/wholeFOV/centered_5steps_stepsize8/2019-12-09_02.29.34/stack_3_channel_0/TP_*/RC_rectified/Cam_Right_001_rectified.tif",
    "LF_computed/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish1/static/wholeFOV/centered_5steps_stepsize8/2019-12-09_02.29.34/stack_3_channel_0/TP_*/RCout/Cam_Right_001.tif",
    "fish1_20191209_0229_static_lr",
)

fish1_20191209_0235_static = NamedDatasetInfo(
    Path(
        "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish1/static/wholeFOV/centered_5steps_stepsize8/2019-12-09_02.35.49"
    ),
    "stack_3_channel_0/TP_*/RC_rectified/Cam_Right_001_rectified.tif",
    "stack_4_channel_1/TP_*/LC/Cam_Left.tif",
    "fish1_20191209_0235_static",
)
fish1_20191209_0235_static_affine = NamedDatasetInfo(
    Path(
        "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish1/static/wholeFOV/centered_5steps_stepsize8/2019-12-09_02.35.49"
    ),
    "stack_3_channel_0/TP_*/RC_rectified/Cam_Right_001_rectified.tif",
    "stack_4_channel_1/TP_*/LC/Cam_Left.tif",
    "fish1_20191209_0235_static_affine",
    AffineTransform=wholeFOV_Transform,
)
fish1_20191209_0235_static_lr = NamedDatasetInfo(
    Path("/g/kreshuk"),
    "LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish1/static/wholeFOV/centered_5steps_stepsize8/2019-12-09_02.35.49/stack_3_channel_0/TP_*/RC_rectified/Cam_Right_001_rectified.tif",
    "LF_computed/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish1/static/wholeFOV/centered_5steps_stepsize8/2019-12-09_02.35.49/stack_3_channel_0/TP_*/RCout/Cam_Right_001.tif",
    "fish1_20191209_0235_static_lr",
)

fish1_20191209_0242_static = NamedDatasetInfo(
    Path(
        "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish1/static/wholeFOV/centered_5steps_stepsize8/2019-12-09_02.42.03"
    ),
    "stack_3_channel_0/TP_*/RC_rectified/Cam_Right_001_rectified.tif",
    "stack_4_channel_1/TP_*/LC/Cam_Left.tif",
    "fish1_20191209_0242_static",
)
fish1_20191209_0242_static_affine = NamedDatasetInfo(
    Path(
        "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish1/static/wholeFOV/centered_5steps_stepsize8/2019-12-09_02.42.03"
    ),
    "stack_3_channel_0/TP_*/RC_rectified/Cam_Right_001_rectified.tif",
    "stack_4_channel_1/TP_*/LC/Cam_Left.tif",
    "fish1_20191209_0242_static_affine",
    AffineTransform=wholeFOV_Transform,
)
fish1_20191209_0242_static_lr = NamedDatasetInfo(
    Path("/g/kreshuk"),
    "LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish1/static/wholeFOV/centered_5steps_stepsize8/2019-12-09_02.42.03/stack_3_channel_0/TP_*/RC_rectified/Cam_Right_001_rectified.tif",
    "LF_computed/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish1/static/wholeFOV/centered_5steps_stepsize8/2019-12-09_02.42.03/stack_3_channel_0/TP_*/RCout/Cam_Right_001.tif",
    "fish1_20191209_0242_static_lr",
)

fish1_20191209_0248_static = NamedDatasetInfo(
    Path(
        "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish1/static/wholeFOV/centered_5steps_stepsize8/2019-12-09_02.48.24"
    ),
    "stack_3_channel_0/TP_*/RC_rectified/Cam_Right_001_rectified.tif",
    "stack_4_channel_1/TP_*/LC/Cam_Left.tif",
    "fish1_20191209_0248_static",
)
fish1_20191209_0248_static_affine = NamedDatasetInfo(
    Path(
        "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish1/static/wholeFOV/centered_5steps_stepsize8/2019-12-09_02.48.24"
    ),
    "stack_3_channel_0/TP_*/RC_rectified/Cam_Right_001_rectified.tif",
    "stack_4_channel_1/TP_*/LC/Cam_Left.tif",
    "fish1_20191209_0248_static_affine",
    AffineTransform=wholeFOV_Transform,
)
fish1_20191209_0248_static_lr = NamedDatasetInfo(
    Path("/g/kreshuk"),
    "LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish1/static/wholeFOV/centered_5steps_stepsize8/2019-12-09_02.48.24/stack_3_channel_0/TP_*/RC_rectified/Cam_Right_001_rectified.tif",
    "LF_computed/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish1/static/wholeFOV/centered_5steps_stepsize8/2019-12-09_02.48.24/stack_3_channel_0/TP_*/RCout/Cam_Right_001.tif",
    "fish1_20191209_0248_static_lr",
)

fish1_20191209_0254_static = NamedDatasetInfo(
    Path(
        "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish1/static/wholeFOV/centered_5steps_stepsize8/2019-12-09_02.54.46"
    ),
    "stack_3_channel_0/TP_*/RC_rectified/Cam_Right_001_rectified.tif",
    "stack_4_channel_1/TP_*/LC/Cam_Left.tif",
    "fish_20191209_0254_static",
)
fish1_20191209_0254_static_affine = NamedDatasetInfo(
    Path(
        "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish1/static/wholeFOV/centered_5steps_stepsize8/2019-12-09_02.54.46"
    ),
    "stack_3_channel_0/TP_*/RC_rectified/Cam_Right_001_rectified.tif",
    "stack_4_channel_1/TP_*/LC/Cam_Left.tif",
    "fish_20191209_0254_static_affine",
    AffineTransform=wholeFOV_Transform,
)

fish1_20191209_0254_static_lr = NamedDatasetInfo(
    Path("/g/kreshuk"),
    "LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish1/static/wholeFOV/centered_5steps_stepsize8/2019-12-09_02.54.46/stack_3_channel_0/TP_*/RC_rectified/Cam_Right_001_rectified.tif",
    "LF_computed/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish1/static/wholeFOV/centered_5steps_stepsize8/2019-12-09_02.54.46/stack_3_channel_0/TP_*/RCout/Cam_Right_001.tif",
    "fish_20191209_0254_static_lr",
)
