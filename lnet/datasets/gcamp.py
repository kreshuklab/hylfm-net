from .base import TensorInfo

ref0_lf = TensorInfo(
    name="lf",
    root="lnet",
    location="ref_data/AffineTransforms/SwipeThrough_-450_-210_nimages_241/*Cam_Right.tif",
    transformations=[],
    meta={},
    insert_singleton_axes_at=[0],
)
ref0_lf_repeated = TensorInfo(
    name="lf",
    root="lnet",
    location="ref_data/AffineTransforms/SwipeThrough_-450_-210_nimages_241/*Cam_Right.tif",
    transformations=[],
    meta={},
    insert_singleton_axes_at=[0],
    repeat=241,
)
ref0_sample_ls_slice = TensorInfo(
    name="ls",
    root="lnet",
    location="ref_data/AffineTransforms/SwipeThrough_-450_-210_nimages_241/*Cam_Left.tif",
    transformations=[],
    meta={},
    insert_singleton_axes_at=[0, 0],
    in_batches_of=241,
    z_slice="idx%241",
)

ref0_lr = TensorInfo(
    name="lr",
    root="lnet",
    location="ref_data/AffineTransforms/SwipeThrough_-450_-210_nimages_241/Gcamp_dataset.h5/t[0-9]+/s00/0/cells",
    transformations=[],
    meta={},
    insert_singleton_axes_at=[0, 0],
)
ref0_ls = TensorInfo(
    name="ls",
    root="lnet",
    location="ref_data/AffineTransforms/SwipeThrough_-450_-210_nimages_241/Gcamp_dataset.h5/t[0-9]+/s01/0/cells",
    transformations=[],
    meta={},
    insert_singleton_axes_at=[0, 0, 0],
)

g200311_085233_lf = TensorInfo(
    name="lf",
    root="GKRESHUK",
    location="LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish2/10Hz/241Planes/2020-03-11_08.52.33/stack_1_channel_3/SwipeThrough_-450_-210_nimages_241/TP_*/RC_rectified/Cam_Right_*_rectified.tif",
    transformations=[],
    meta={},
    insert_singleton_axes_at=[0, 0],
    # length=31*241,
)
g200311_085233_ls = TensorInfo(
    name="ls",
    root="GKRESHUK",
    location="LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish2/10Hz/241Planes/2020-03-11_08.52.33/stack_1_channel_3/SwipeThrough_-450_-210_nimages_241/TP_*/LC/Cam_Left_*.tif",
    transformations=[],
    meta={},
    insert_singleton_axes_at=[0, 0, 0],
    z_slice="idx%241",
    # length=31*241,
)

g200311_090800_lf = TensorInfo(
    name="lf",
    root="GKRESHUK",
    location="LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish2/10Hz/241Planes/2020-03-11_09.08.00/stack_1_channel_3/SwipeThrough_-450_-210_nimages_241/TP_*/RC_rectified/Cam_Right_*_rectified.tif",
    transformations=[],
    meta={},
    insert_singleton_axes_at=[0, 0],
    # length=15*241,
)
g200311_090800_ls = TensorInfo(
    name="ls",
    root="GKRESHUK",
    location="LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish2/10Hz/241Planes/2020-03-11_09.08.00/stack_1_channel_3/SwipeThrough_-450_-210_nimages_241/TP_*/LC/Cam_Left_*.tif",
    transformations=[],
    meta={},
    insert_singleton_axes_at=[0, 0, 0],
    z_slice="idx%241"
    # length=15*241,
)

# 7 * 8 x 121 planes
# TestOutputGcamp\LenseLeNet_Microscope\20200311_Gcamp\fish2\10Hz\slideThrough
g200311_083021_lf = TensorInfo(
    name="lf",
    root="GKRESHUK",
    location="LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish2/10Hz/slideThrough/2020-03-11_08.30.21/stack_36_channel_3/SwipeThrough_-390_-270_nimages_121/TP_*/RC_rectified/Cam_Right_*_rectified.tif",
    transformations=[],
    meta={},
    insert_singleton_axes_at=[0, 0],
    # length=8*121,
)
g200311_083021_ls = TensorInfo(
    name="ls",
    root="GKRESHUK",
    location="LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish2/10Hz/241Planes/2020-03-11_09.08.00/stack_1_channel_3/SwipeThrough_-450_-210_nimages_241/TP_*/LC/Cam_Left_*.tif",
    transformations=[],
    meta={},
    insert_singleton_axes_at=[0, 0, 0],
    z_slice="60+idx%121"
    # length=8*121,
)

g200311_083419_lf = TensorInfo(
    name="lf",
    root="GKRESHUK",
    location="LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish2/10Hz/slideThrough/2020-03-11_08.34.19/stack_36_channel_3/SwipeThrough_-390_-270_nimages_121/TP_*/RC_rectified/Cam_Right_*_rectified.tif",
    transformations=[],
    meta={},
    insert_singleton_axes_at=[0, 0],
    # length=8*121,
)
g200311_083419_ls = TensorInfo(
    name="ls",
    root="GKRESHUK",
    location="LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish2/10Hz/241Planes/2020-03-11_08.34.19/stack_1_channel_3/SwipeThrough_-450_-210_nimages_241/TP_*/LC/Cam_Left_*.tif",
    transformations=[],
    meta={},
    insert_singleton_axes_at=[0, 0, 0],
    z_slice="60+idx%121"
    # length=8*121,
)

g200311_083658_lf = TensorInfo(
    name="lf",
    root="GKRESHUK",
    location="LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish2/10Hz/slideThrough/2020-03-11_08.36.58/stack_36_channel_3/SwipeThrough_-390_-270_nimages_121/TP_*/RC_rectified/Cam_Right_*_rectified.tif",
    transformations=[],
    meta={},
    insert_singleton_axes_at=[0, 0],
    # length=8*121,
)
g200311_083658_ls = TensorInfo(
    name="ls",
    root="GKRESHUK",
    location="LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish2/10Hz/241Planes/2020-03-11_08.36.58/stack_1_channel_3/SwipeThrough_-450_-210_nimages_241/TP_*/LC/Cam_Left_*.tif",
    transformations=[],
    meta={},
    insert_singleton_axes_at=[0, 0, 0],
    z_slice="60+idx%121"
    # length=8*121,
)

g200311_083931_lf = TensorInfo(
    name="lf",
    root="GKRESHUK",
    location="LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish2/10Hz/slideThrough/2020-03-11_08.39.31/stack_36_channel_3/SwipeThrough_-390_-270_nimages_121/TP_*/RC_rectified/Cam_Right_*_rectified.tif",
    transformations=[],
    meta={},
    insert_singleton_axes_at=[0, 0],
    # length=8*121,
)
g200311_083931_ls = TensorInfo(
    name="ls",
    root="GKRESHUK",
    location="LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish2/10Hz/241Planes/2020-03-11_08.39.31/stack_1_channel_3/SwipeThrough_-450_-210_nimages_241/TP_*/LC/Cam_Left_*.tif",
    transformations=[],
    meta={},
    insert_singleton_axes_at=[0, 0, 0],
    z_slice="60+idx%121"
    # length=8*121,
)

g200311_084207_lf = TensorInfo(
    name="lf",
    root="GKRESHUK",
    location="LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish2/10Hz/slideThrough/2020-03-11_08.42.07/stack_36_channel_3/SwipeThrough_-390_-270_nimages_121/TP_*/RC_rectified/Cam_Right_*_rectified.tif",
    transformations=[],
    meta={},
    insert_singleton_axes_at=[0, 0],
    # length=8*121,
)
g200311_084207_ls = TensorInfo(
    name="ls",
    root="GKRESHUK",
    location="LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish2/10Hz/241Planes/2020-03-11_08.42.07/stack_1_channel_3/SwipeThrough_-450_-210_nimages_241/TP_*/LC/Cam_Left_*.tif",
    transformations=[],
    meta={},
    insert_singleton_axes_at=[0, 0, 0],
    z_slice="60+idx%121"
    # length=8*121,
)

g200311_084450_lf = TensorInfo(
    name="lf",
    root="GKRESHUK",
    location="LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish2/10Hz/slideThrough/2020-03-11_08.44.50/stack_36_channel_3/SwipeThrough_-390_-270_nimages_121/TP_*/RC_rectified/Cam_Right_*_rectified.tif",
    transformations=[],
    meta={},
    insert_singleton_axes_at=[0, 0],
    # length=8*121,
)
g200311_084450_ls = TensorInfo(
    name="ls",
    root="GKRESHUK",
    location="LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish2/10Hz/241Planes/2020-03-11_08.44.50/stack_1_channel_3/SwipeThrough_-450_-210_nimages_241/TP_*/LC/Cam_Left_*.tif",
    transformations=[],
    meta={},
    insert_singleton_axes_at=[0, 0, 0],
    z_slice="60+idx%121"
    # length=8*121,
)

g200311_084729_lf = TensorInfo(
    name="lf",
    root="GKRESHUK",
    location="LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish2/10Hz/slideThrough/2020-03-11_08.47.29/stack_36_channel_3/SwipeThrough_-390_-270_nimages_121/TP_*/RC_rectified/Cam_Right_*_rectified.tif",
    transformations=[],
    meta={},
    insert_singleton_axes_at=[0, 0],
    # length=8*121,
)
g200311_084729_ls = TensorInfo(
    name="ls",
    root="GKRESHUK",
    location="LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish2/10Hz/241Planes/2020-03-11_08.47.29/stack_1_channel_3/SwipeThrough_-450_-210_nimages_241/TP_*/LC/Cam_Left_*.tif",
    transformations=[],
    meta={},
    insert_singleton_axes_at=[0, 0, 0],
    z_slice="60+idx%121"
    # length=8*121,
)
