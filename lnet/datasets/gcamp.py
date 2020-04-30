from lnet import settings
from lnet.datasets.base import TensorInfo

ref0_lf = TensorInfo(
    name="lf",
    root="lnet",
    location="ref_data/AffineTransforms/SwipeThrough_-450_-210_nimages_241/*Cam_Right.tif",
    insert_singleton_axes_at=[0],
    tag="ref0",
)
ref0_lf_repeated = TensorInfo(
    name="lf",
    root="lnet",
    location="ref_data/AffineTransforms/SwipeThrough_-450_-210_nimages_241/*Cam_Right.tif",
    insert_singleton_axes_at=[0],
    repeat=241,
    tag="ref0_repeated",
)
ref0_sample_ls_slice = TensorInfo(
    name="ls",
    root="lnet",
    location="ref_data/AffineTransforms/SwipeThrough_-450_-210_nimages_241/*Cam_Left.tif",
    insert_singleton_axes_at=[0, 0],
    datasets_per_file=241,
    z_slice="idx%241",
    tag="ref0_slice",
)

ref0_lr = TensorInfo(
    name="lr",
    root="lnet",
    location="ref_data/AffineTransforms/SwipeThrough_-450_-210_nimages_241/Gcamp_dataset.h5/t[0-9]+/s00/0/cells",
    insert_singleton_axes_at=[0, 0],
    tag="ref0",
)
ref0_ls = TensorInfo(
    name="ls",
    root="lnet",
    location="ref_data/AffineTransforms/SwipeThrough_-450_-210_nimages_241/Gcamp_dataset.h5/t[0-9]+/s01/0/cells",
    insert_singleton_axes_at=[0, 0, 0],
    tag="ref0",
)

# TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish2/10Hz/241Planes
g200311_085233_lf = TensorInfo(
    name="lf",
    root="GKRESHUK",
    location="LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish2/10Hz/241Planes/2020-03-11_08.52.33/stack_1_channel_3/SwipeThrough_-450_-210_nimages_241/TP_*/RC_rectified/Cam_Right_*_rectified.tif",
    insert_singleton_axes_at=[0, 0],
    tag="g200311_085233",
    # length=30*241=7230,
)
g200311_085233_ls = TensorInfo(
    name="ls",
    root="GKRESHUK",
    location="LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish2/10Hz/241Planes/2020-03-11_08.52.33/stack_1_channel_3/SwipeThrough_-450_-210_nimages_241/TP_*/LC/Cam_Left_*.tif",
    insert_singleton_axes_at=[0, 0, 0],
    z_slice="idx%241",
    tag="g200311_085233",
    # length=30*241=7230,
)

g200311_090800_lf = TensorInfo(
    name="lf",
    root="GKRESHUK",
    location="LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish2/10Hz/241Planes/2020-03-11_09.08.00/stack_1_channel_3/SwipeThrough_-450_-210_nimages_241/TP_*/RC_rectified/Cam_Right_*_rectified.tif",
    insert_singleton_axes_at=[0, 0],
    tag="g200311_090800",
    # length=15*241,
)
g200311_090800_ls = TensorInfo(
    name="ls",
    root="GKRESHUK",
    location="LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish2/10Hz/241Planes/2020-03-11_09.08.00/stack_1_channel_3/SwipeThrough_-450_-210_nimages_241/TP_*/LC/Cam_Left_*.tif",
    insert_singleton_axes_at=[0, 0, 0],
    tag="g200311_090800",
    z_slice="idx%241"
    # length=15*241,
)

# 7 * 8 x 121 planes
# TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish2/10Hz/slideThrough
g200311_083021_lf = TensorInfo(
    name="lf",
    root="GKRESHUK",
    location="LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish2/10Hz/slideThrough/2020-03-11_08.30.21/stack_36_channel_3/SwipeThrough_-390_-270_nimages_121/TP_*/RC_rectified/Cam_Right_*_rectified.tif",
    insert_singleton_axes_at=[0, 0],
    tag="g200311_083021",
    # length=8*121,
)
g200311_083021_ls = TensorInfo(
    name="ls",
    root="GKRESHUK",
    location="LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish2/10Hz/slideThrough/2020-03-11_08.30.21/stack_36_channel_3/SwipeThrough_-390_-270_nimages_121/TP_*/LC/Cam_Left_*.tif",
    insert_singleton_axes_at=[0, 0, 0],
    tag="g200311_083021",
    z_slice="60+idx%121"
    # length=8*121,
)

g200311_083419_lf = TensorInfo(
    name="lf",
    root="GKRESHUK",
    location="LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish2/10Hz/slideThrough/2020-03-11_08.34.19/stack_36_channel_3/SwipeThrough_-390_-270_nimages_121/TP_*/RC_rectified/Cam_Right_*_rectified.tif",
    insert_singleton_axes_at=[0, 0],
    tag="g200311_083419",
    # length=8*121,
)
g200311_083419_ls = TensorInfo(
    name="ls",
    root="GKRESHUK",
    location="LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish2/10Hz/slideThrough/2020-03-11_08.34.19/stack_36_channel_3/SwipeThrough_-390_-270_nimages_121/TP_*/LC/Cam_Left_*.tif",
    insert_singleton_axes_at=[0, 0, 0],
    tag="g200311_083419",
    z_slice="60+idx%121"
    # length=8*121,
)

g200311_083658_lf = TensorInfo(
    name="lf",
    root="GKRESHUK",
    location="LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish2/10Hz/slideThrough/2020-03-11_08.36.58/stack_36_channel_3/SwipeThrough_-390_-270_nimages_121/TP_*/RC_rectified/Cam_Right_*_rectified.tif",
    insert_singleton_axes_at=[0, 0],
    tag="g200311_083658",
    # length=8*121,
)
g200311_083658_ls = TensorInfo(
    name="ls",
    root="GKRESHUK",
    location="LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish2/10Hz/slideThrough/2020-03-11_08.36.58/stack_36_channel_3/SwipeThrough_-390_-270_nimages_121/TP_*/LC/Cam_Left_*.tif",
    insert_singleton_axes_at=[0, 0, 0],
    tag="g200311_083658",
    z_slice="60+idx%121"
    # length=8*121,
)

g200311_083931_lf = TensorInfo(
    name="lf",
    root="GKRESHUK",
    location="LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish2/10Hz/slideThrough/2020-03-11_08.39.31/stack_36_channel_3/SwipeThrough_-390_-270_nimages_121/TP_*/RC_rectified/Cam_Right_*_rectified.tif",
    insert_singleton_axes_at=[0, 0],
    tag="g200311_083931",
    # length=8*121,
)
g200311_083931_ls = TensorInfo(
    name="ls",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20200311_Gcamp/fish2/10Hz/slideThrough/2020-03-11_08.39.31/stack_36_channel_3/Cam_Left_*.h5/Data",
    insert_singleton_axes_at=[0, 0],
    tag="g200311_083931",
    z_slice="60+idx%121",
    samples_per_dataset=121,
    # length=8*121,
)

g200311_084207_lf = TensorInfo(
    name="lf",
    root="GKRESHUK",
    location="LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish2/10Hz/slideThrough/2020-03-11_08.42.07/stack_36_channel_3/SwipeThrough_-390_-270_nimages_121/TP_*/RC_rectified/Cam_Right_*_rectified.tif",
    insert_singleton_axes_at=[0, 0],
    tag="g200311_084207",
    # length=8*121,
)
g200311_084207_ls = TensorInfo(
    name="ls",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20200311_Gcamp/fish2/10Hz/slideThrough/2020-03-11_08.42.07/stack_36_channel_3/Cam_Left_*.h5/Data",
    insert_singleton_axes_at=[0, 0],
    tag="g200311_084207",
    z_slice="60+idx%121",
    samples_per_dataset=121,
    # length=8*121,
)

g200311_084450_lf = TensorInfo(
    name="lf",
    root="GKRESHUK",
    location="LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish2/10Hz/slideThrough/2020-03-11_08.44.50/stack_36_channel_3/SwipeThrough_-390_-270_nimages_121/TP_*/RC_rectified/Cam_Right_*_rectified.tif",
    insert_singleton_axes_at=[0, 0],
    tag="g200311_084450",
    # length=8*121,
)
g200311_084450_ls = TensorInfo(
    name="ls",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20200311_Gcamp/fish2/10Hz/slideThrough/2020-03-11_08.44.50/stack_36_channel_3/Cam_Left_*.h5/Data",
    insert_singleton_axes_at=[0, 0],
    tag="g200311_084450",
    z_slice="60+idx%121",
    samples_per_dataset=121,
    # length=8*121,
)

g200311_084729_lf = TensorInfo(
    name="lf",
    root="GKRESHUK",
    location="LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish2/10Hz/slideThrough/2020-03-11_08.47.29/stack_36_channel_3/SwipeThrough_-390_-270_nimages_121/TP_*/RC_rectified/Cam_Right_*_rectified.tif",
    insert_singleton_axes_at=[0, 0],
    tag="g200311_084729",
    # length=8*121,
)
g200311_084729_ls = TensorInfo(
    name="ls",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20200311_Gcamp/fish2/10Hz/slideThrough/2020-03-11_08.47.29/stack_36_channel_3/Cam_Left_*.h5/Data",
    insert_singleton_axes_at=[0, 0],
    tag="g200311_084729",
    z_slice="60+idx%121",
    samples_per_dataset=121,
    # length=8*121,
)


# single plane TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish2/10Hz/singlePlane
g20200311_073039_70_lf = TensorInfo(
    name="lf",
    root="GKRESHUK",
    location="LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish2/10Hz/singlePlane/2020-03-11_07.30.39/stack_7_channel_3/SinglePlane_-320/TP_*/RC_rectified/Cam_Right_*_rectified.tif",
    insert_singleton_axes_at=[0, 0],
    tag="g200311_073039_70",
    # length=600,
)
g20200311_073039_70_ls = TensorInfo(
    name="ls",
    root="GKRESHUK",
    location="LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish2/10Hz/singlePlane/2020-03-11_07.30.39/stack_7_channel_3/SinglePlane_-320/TP_*/LC/Cam_Left_*.tif",
    insert_singleton_axes_at=[0, 0, 0],
    z_slice=70,
    tag="g200311_073039_70",
    # length=600,
)


g20200311_073039_80_lf = TensorInfo(
    name="lf",
    root="GKRESHUK",
    location="LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish2/10Hz/singlePlane/2020-03-11_07.30.39/stack_8_channel_3/SinglePlane_-310/TP_*/RC_rectified/Cam_Right_*_rectified.tif",
    insert_singleton_axes_at=[0, 0],
    tag="g200311_073039_80",
    # length=600,
)
g20200311_073039_80_ls = TensorInfo(
    name="ls",
    root="GKRESHUK",
    location="LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish2/10Hz/singlePlane/2020-03-11_07.30.39/stack_8_channel_3/SinglePlane_-310/TP_*/LC/Cam_Left_*.tif",
    insert_singleton_axes_at=[0, 0, 0],
    z_slice=80,
    tag="g200311_073039_80",
    # length=600,
)


g20200311_101320_100_lf = TensorInfo(
    name="lf",
    root="GKRESHUK",
    location="LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish2/10Hz/singlePlane/2020-03-11_10.13.20/stack_10_channel_3/SinglePlane_-290/TP_*/RC_rectified/Cam_Right_*_rectified.tif",
    insert_singleton_axes_at=[0, 0],
    tag="g20200311_101320_100",
    # length=
)
g20200311_101320_100_ls = TensorInfo(
    name="ls",
    root="GKRESHUK",
    location="LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish2/10Hz/singlePlane/2020-03-11_10.13.20/stack_10_channel_3/SinglePlane_-290/TP_*/LC/Cam_Left_*.tif",
    insert_singleton_axes_at=[0, 0, 0],
    z_slice=100,
    tag="00311_101320_100",
    # length=600
)


g20200311_101320_75_lf = TensorInfo(
    name="lf",
    root="GKRESHUK",
    location="LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish2/10Hz/singlePlane/2020-03-11_10.13.20/stack_24_channel_3/SinglePlane_-315/TP_*/RC_rectified/Cam_Right_*_rectified.tif",
    insert_singleton_axes_at=[0, 0],
    tag="g200311_101320_75",
    # length=
)
g20200311_101320_75_ls = TensorInfo(
    name="ls",
    root="GKRESHUK",
    location="LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish2/10Hz/singlePlane/2020-03-11_10.13.20/stack_24_channel_3/SinglePlane_-315/TP_*/LC/Cam_Left_*.tif",
    insert_singleton_axes_at=[0, 0, 0],
    z_slice=75,
    tag="g200311_101320_75",
    # length=600
)


g20200311_102541_50_lf = TensorInfo(
    name="lf",
    root="GKRESHUK",
    location="LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish2/10Hz/singlePlane/2020-03-11_10.25.41/stack_3_channel_3/SinglePlane_-340/TP_*/RC_rectified/Cam_Right_*_rectified.tif",
    insert_singleton_axes_at=[0, 0],
    tag="g200311_102541_50",
    # length=
)
g20200311_102541_50_ls = TensorInfo(
    name="ls",
    root="GKRESHUK",
    location="LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish2/10Hz/singlePlane/2020-03-11_10.25.41/stack_3_channel_3/SinglePlane_-340/TP_*/LC/Cam_Left_*.tif",
    insert_singleton_axes_at=[0, 0, 0],
    z_slice=50,
    tag="g200311_102541_50",
    # length=600
)

# g20200311_102541_95_lf = TensorInfo(
#     name="lf",
#     root="GKRESHUK",
#     location="LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish2/10Hz/singlePlane/2020-03-11_10.25.41/stack_25_channel_3/SinglePlane_-295/TP_*/RC_rectified/Cam_Right_*_rectified.tif",
#     insert_singleton_axes_at=[0, 0],
#     tag="g200311_102541_95",
#     # length=600
# )
# g20200311_102541_95_ls = TensorInfo(
#     name="ls",
#     root="GKRESHUK",
#     location="LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish2/10Hz/singlePlane/2020-03-11_10.25.41/stack_25_channel_3/SinglePlane_-295/TP_*/LC/Cam_Left_*.tif",
#     insert_singleton_axes_at=[0, 0, 0],
#     z_slice=95,
#     tag="g200311_102541_95",
#     # length=
# )
g20200311_102541_26_lf = TensorInfo(
    name="lf",
    root="GKRESHUK",
    location="LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish2/10Hz/singlePlane/2020-03-11_10.25.41/stack_25_channel_3/SinglePlane_-295/TP_*/RC_rectified/Cam_Right_*_rectified.tif",
    insert_singleton_axes_at=[0, 0],
    tag="g200311_102541_26",
    # length=600
)
g20200311_102541_26_ls = TensorInfo(
    name="ls",
    root="GKRESHUK",
    location="LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish2/10Hz/singlePlane/2020-03-11_10.25.41/stack_25_channel_3/SinglePlane_-295/TP_*/LC/Cam_Left_*.tif",
    insert_singleton_axes_at=[0, 0, 0],
    z_slice=26,
    tag="g200311_102541_26",
    # length=
)


g20200311_102541_85_lf = TensorInfo(
    name="lf",
    root="GKRESHUK",
    location="LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish2/10Hz/singlePlane/2020-03-11_10.25.41/stack_29_channel_3/SinglePlane_-305/TP_*/RC_rectified/Cam_Right_*_rectified.tif",
    insert_singleton_axes_at=[0, 0],
    tag="g200311_102541_85",
    # length=
)
g20200311_102541_85_ls = TensorInfo(
    name="ls",
    root="GKRESHUK",
    location="LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish2/10Hz/singlePlane/2020-03-11_10.25.41/stack_29_channel_3/SinglePlane_-305/TP_*/LC/Cam_Left_*.tif",
    insert_singleton_axes_at=[0, 0, 0],
    z_slice=85,
    tag="g200311_102541_85",
    # length=600
)


# -390 to -270, 121 planes:
g20200311_065726_lf = TensorInfo(
    name="lf",
    root="GKRESHUK",
    location="LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish2/5Hz/2020-03-11_06.57.26/stack_33_channel_4/SwipeThrough_-390_-270_nimages_121/TP_*/RC_rectified/Cam_Right_*_rectified.tif",
    insert_singleton_axes_at=[0, 0],
    tag="g20200311_065726",
    # length=
)
g20200311_065726_ls = TensorInfo(
    name="ls",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20200311_Gcamp/fish2/5Hz/2020-03-11_06.57.26/stack_33_channel_4/Cam_Left_*.h5/Data",
    insert_singleton_axes_at=[0, 0],
    tag="g20200311_065726",
    z_slice="60+idx%121",
    samples_per_dataset=121,
    # length=6534
)


g20200311_073447_lf = TensorInfo(
    name="lf",
    root="GKRESHUK",
    location="LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish2/10Hz/2020-03-11_07.34.47/stack_33_channel_3/SwipeThrough_-390_-270_nimages_121/TP_*/RC_rectified/Cam_Right_*_rectified.tif",
    insert_singleton_axes_at=[0, 0],
    tag="g20200311_073447",
    # length=
)
g20200311_073447_ls = TensorInfo(
    name="ls",
    root="GKRESHUK",
    location="LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish2/10Hz/2020-03-11_07.34.47/stack_33_channel_3/SwipeThrough_-390_-270_nimages_121/TP_*/LC/Cam_Left_*.tif",
    insert_singleton_axes_at=[0, 0, 0],
    tag="g20200311_073447",
    z_slice="60+idx%121",
    # length=9680
)


g20200311_033035_lf = TensorInfo(
    name="lf",
    root="GKRESHUK",
    location="LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish1/5Hz/longRun/niceOne/2020-03-11_03.30.35/stack_33_channel_4/SwipeThrough_-390_-270_nimages_121/TP_*/RC_rectified/Cam_Right_*_rectified.tif",
    insert_singleton_axes_at=[0, 0],
    tag="g20200311_033035",
    # length=
)
g20200311_033035_ls = TensorInfo(
    name="ls",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20200311_Gcamp/fish1/5Hz/longRun/niceOne/2020-03-11_03.30.35/stack_33_channel_4/Cam_Left_*.h5/Data",
    insert_singleton_axes_at=[0, 0],
    tag="g20200311_033035",
    z_slice="60+idx%121",
    samples_per_dataset=121,
    # length=1331
)


g20200311_040327_lf = TensorInfo(
    name="lf",
    root="GKRESHUK",
    location="LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish1/5Hz/longRun/niceOne2/2020-03-11_04.03.27/stack_33_channel_4/SwipeThrough_-390_-270_nimages_121/TP_*/RC_rectified/Cam_Right_*_rectified.tif",
    insert_singleton_axes_at=[0, 0],
    tag="g20200311_040327",
    z_slice="60+idx%121",
    # length=
)
g20200311_040327_ls = TensorInfo(
    name="ls",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20200311_Gcamp/fish1/5Hz/longRun/niceOne2/2020-03-11_04.03.27/stack_33_channel_4/Cam_Left_*.h5/Data",
    insert_singleton_axes_at=[0, 0],
    tag="g20200311_040327",
    z_slice="60+idx%121",
    samples_per_dataset=121,
    # length=
)


g20200311_042734_lf = TensorInfo(
    name="lf",
    root="GKRESHUK",
    location="LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish1/5Hz/longRun/niceOne3/2020-03-11_04.27.34/stack_33_channel_4/SwipeThrough_-390_-270_nimages_121/TP_*/RC_rectified/Cam_Right_*_rectified.tif",
    insert_singleton_axes_at=[0, 0],
    tag="g20200311_042734",
    z_slice="60+idx%121",
    # length=
)
g20200311_042734_ls = TensorInfo(
    name="ls",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20200311_Gcamp/fish1/5Hz/longRun/niceOne3/2020-03-11_04.27.34/stack_33_channel_4/Cam_Left_*.h5/Data",
    insert_singleton_axes_at=[0, 0],
    tag="g20200311_042734",
    z_slice="60+idx%121",
    samples_per_dataset=121,
    # length=
)


g20200311_042839_lf = TensorInfo(
    name="lf",
    root="GKRESHUK",
    location="LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish1/5Hz/longRun/niceOne3/2020-03-11_04.28.39/stack_33_channel_4/SwipeThrough_-390_-270_nimages_121/TP_*/RC_rectified/Cam_Right_*_rectified.tif",
    insert_singleton_axes_at=[0, 0],
    tag="g20200311_042839",
    z_slice="60+idx%121",
    # length=
)
g20200311_042839_ls = TensorInfo(
    name="ls",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20200311_Gcamp/fish1/5Hz/longRun/niceOne3/2020-03-11_04.28.39/stack_33_channel_4/Cam_Left_*.h5/Data",
    insert_singleton_axes_at=[0, 0],
    tag="g20200311_042839",
    z_slice="60+idx%121",
    samples_per_dataset=121,
    # length=
)


g20200311_043502_lf = TensorInfo(
    name="lf",
    root="GKRESHUK",
    location="LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish1/5Hz/longRun/niceOne3/2020-03-11_04.35.02/stack_33_channel_4/SwipeThrough_-390_-270_nimages_121/TP_*/RC_rectified/Cam_Right_*_rectified.tif",
    insert_singleton_axes_at=[0, 0],
    z_slice="60+idx%121",
    tag="g20200311_043502",
    # length=
)
g20200311_043502_ls = TensorInfo(
    name="ls",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20200311_Gcamp/fish1/5Hz/longRun/niceOne3/2020-03-11_04.35.02/stack_33_channel_4/Cam_Left_*.h5/Data",
    insert_singleton_axes_at=[0, 0],
    tag="g20200311_043502",
    z_slice="60+idx%121",
    samples_per_dataset=121,
    # length=
)


g20200311_024927_lf = TensorInfo(
    name="lf",
    root="GKRESHUK",
    location="LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish1/5Hz/longRun/2020-03-11_02.49.27/stack_33_channel_8/SwipeThrough_-390_-270_nimages_121/TP_*/RC_rectified/Cam_Right_*_rectified.tif",
    insert_singleton_axes_at=[0, 0],
    z_slice="60+idx%121",
    tag="g20200311_024927",
    # length=
)
g20200311_024927_ls = TensorInfo(
    name="ls",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20200311_Gcamp/fish1/5Hz/longRun/2020-03-11_02.49.27/stack_33_channel_8/Cam_Left_*.h5/Data",
    insert_singleton_axes_at=[0, 0],
    tag="g20200311_024927",
    z_slice="60+idx%121",
    samples_per_dataset=121,
    # length=
)


# single plane super nice (for calcium traces)
g20200311_101734_110_lf = TensorInfo(
    name="lf",
    root="GKRESHUK",
    location="LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish2/10Hz/singlePlane/superNice/2020-03-11_10.17.34/stack_14_channel_3/SinglePlane_-280/TP_*/RC_rectified/Cam_Right_*_rectified.tif",
    insert_singleton_axes_at=[0, 0],
    tag="00311_101734_110",
    # length=600
)
g20200311_101734_110_ls = TensorInfo(
    name="ls",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20200311_Gcamp/fish2/10Hz/singlePlane/superNice/2020-03-11_10.17.34/stack_14_channel_3/Cam_Left_*.h5/Data",
    insert_singleton_axes_at=[0, 0],
    z_slice=110,
    tag="00311_101734_110",
    samples_per_dataset=600,
    # length=600
)

g20200311_101734_60_lf = TensorInfo(
    name="lf",
    root="GKRESHUK",
    location="LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish2/10Hz/singlePlane/superNice/2020-03-11_10.17.34/stack_16_channel_3/SinglePlane_-330/TP_*/RC_rectified/Cam_Right_*_rectified.tif",
    insert_singleton_axes_at=[0, 0],
    tag="g200311_101734_60",
    # length=600
)
g20200311_101734_60_ls = TensorInfo(
    name="ls",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20200311_Gcamp/fish2/10Hz/singlePlane/superNice/2020-03-11_10.17.34/stack_16_channel_3/Cam_Left_*.h5/Data",
    insert_singleton_axes_at=[0, 0],
    z_slice=60,
    tag="g200311_101734_60",
    samples_per_dataset=600,
    # length=600
)

# g20200311_102114_95_lf = TensorInfo(
#     name="lf",
#     root="GKRESHUK",
#     location="LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish2/10Hz/singlePlane/superNice/2020-03-11_10.21.14/stack_25_channel_3/SinglePlane_-295/TP_*/RC_rectified/Cam_Right_*_rectified.tif",
#     insert_singleton_axes_at=[0, 0],
#     tag="g200311_102114_95",
#     # length=600
# )
# g20200311_102114_95_ls = TensorInfo(
#     name="ls",
#     root="GKRESHUK",
#     location="LF_partially_restored/LenseLeNet_Microscope/20200311_Gcamp/fish2/10Hz/singlePlane/superNice/2020-03-11_10.21.14/stack_25_channel_3/Cam_Left_*.h5/Data",
#     insert_singleton_axes_at=[0, 0],
#     z_slice=95,
#     tag="g200311_102114_95",
#     samples_per_dataset=600,
#     # length=600
# )
g20200311_102114_26_lf = TensorInfo(
    name="lf",
    root="GKRESHUK",
    location="LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish2/10Hz/singlePlane/superNice/2020-03-11_10.21.14/stack_25_channel_3/SinglePlane_-295/TP_*/RC_rectified/Cam_Right_*_rectified.tif",
    insert_singleton_axes_at=[0, 0],
    tag="g200311_102114_26",
    # length=600
)
g20200311_102114_26_ls = TensorInfo(
    name="ls",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20200311_Gcamp/fish2/10Hz/singlePlane/superNice/2020-03-11_10.21.14/stack_25_channel_3/Cam_Left_*.h5/Data",
    insert_singleton_axes_at=[0, 0],
    z_slice=26,
    tag="g200311_102114_26",
    samples_per_dataset=600,
    # length=600
)


g20200311_102114_85_lf = TensorInfo(
    name="lf",
    root="GKRESHUK",
    location="LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish2/10Hz/singlePlane/superNice/2020-03-11_10.21.14/stack_29_channel_3/SinglePlane_-305/TP_*/RC_rectified/Cam_Right_*_rectified.tif",
    insert_singleton_axes_at=[0, 0],
    tag="g200311_102114_85",
    # length=600
)
g20200311_102114_85_ls = TensorInfo(
    name="ls",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20200311_Gcamp/fish2/10Hz/singlePlane/superNice/2020-03-11_10.21.14/stack_29_channel_3/Cam_Left_*.h5/Data",
    insert_singleton_axes_at=[0, 0],
    z_slice=85,
    tag="g200311_102114_85",
    samples_per_dataset=600,
    # length=600
)


# -450 to -210, 121 planes
# = TensorInfo(
#     name="lf",
#     root="GKRESHUK",
#     location="LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish1/5Hz/longRun/slideThrough/2020-03-11_05.26.26/stack_35_channel_4/SwipeThrough_-450_-210_nimages_121",
#     insert_singleton_axes_at=[0, 0],
# )
# = TensorInfo(
#     name="lf",
#     root="GKRESHUK",
#     location="LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish1/5Hz/longRun/niceOne5/2020-03-11_05.53.13/stack_32_channel_4/SwipeThrough_-450_-210_nimages_121",
#     insert_singleton_axes_at=[0, 0],
# )
# = TensorInfo(
#     name="lf",
#     root="GKRESHUK",
#     location="LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish1/5Hz/longRun/niceOne4/2020-03-11_04.47.24/stack_32_channel_4/SwipeThrough_-450_-210_nimages_121",
#     insert_singleton_axes_at=[0, 0],
# )
# = TensorInfo(
#     name="lf",
#     root="GKRESHUK",
#     location="LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish2/5Hz/2020-03-11_06.55.38/stack_32_channel_4/SwipeThrough_-450_-210_nimages_121",
#     insert_singleton_axes_at=[0, 0],
# )


# -450 to -210, 49 planes (150 files)
# = TensorInfo(
#     name="lf",
#     root="GKRESHUK",
#     location="LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200309_Gcamp/fish4_promising/longRun/fov2/2020-03-09_09.06.55/stack_28_channel_4/SwipeThrough_-450_-210_nimages_49",
#     insert_singleton_axes_at=[0, 0],
# )
# = TensorInfo(
#     name="lf",
#     root="GKRESHUK",
#     location="LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200309_Gcamp/fish4_promising/longRun/fov1/2020-03-09_08.31.10/stack_28_channel_4/SwipeThrough_-450_-210_nimages_49",
#     insert_singleton_axes_at=[0, 0],
# )
# = TensorInfo(
#     name="lf",
#     root="GKRESHUK",
#     location="LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish2/10Hz/241Planes/2020-03-11_08.52.33/stack_1_channel_3/SwipeThrough_-450_-210_nimages_241",
#     insert_singleton_axes_at=[0, 0],
# )
# = TensorInfo(
#     name="lf",
#     root="GKRESHUK",
#     location="LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish2/10Hz/241Planes/2020-03-11_09.08.00/stack_1_channel_3/SwipeThrough_-450_-210_nimages_241",
#     insert_singleton_axes_at=[0, 0],
# )


# -390 to -270, 25 planes
# = TensorInfo(
#     name="lf",
#     root="GKRESHUK",
#     location="LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200309_Gcamp/fish4_promising/longRun/fov1/2020-03-09_08.20.04/stack_11_channel_3/SwipeThrough_-390_-270_nimages_25",
#     insert_singleton_axes_at=[0, 0],
# )
# = TensorInfo(
#     name="lf",
#     root="GKRESHUK",
#     location="LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200309_Gcamp/fish4_promising/longRun/fov1/2020-03-09_08.21.40/stack_11_channel_3/SwipeThrough_-390_-270_nimages_25",
#     insert_singleton_axes_at=[0, 0],
# )


# all kinds of single planes
# = TensorInfo(
#     name="lf",
#     root="GKRESHUK",
#     location="LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200309_Gcamp/fish4_promising/singlePlanes/fov1/2020-03-09_08.41.22/stack_2_channel_4",
#     insert_singleton_axes_at=[0, 0],
# )
# = TensorInfo(
#     name="lf",
#     root="GKRESHUK",
#     location="LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200309_Gcamp/fish4_promising/singlePlanes/fov1/2020-03-09_08.41.22/...",
#     insert_singleton_axes_at=[0, 0],
# )

g20200309_043555_60_lf = TensorInfo(
    name="lf",
    root="GKRESHUK",
    location="LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200309_Gcamp/fish1_awesome/2020-03-09_04.35.55/stack_2_channel_3/SinglePlane_-330/TP_*/RC_rectified/Cam_Right_*_rectified.tif",
    insert_singleton_axes_at=[0, 0],
    tag="g20200309_043555_60",
    # length=
)
g20200309_043555_60_ls = TensorInfo(
    name="ls",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20200309_Gcamp/fish1_awesome/2020-03-09_04.35.55/stack_2_channel_3/Cam_Left_*.h5/Data",
    insert_singleton_axes_at=[0, 0],
    z_slice=60,
    tag="g20200309_043555_60",
    samples_per_dataset=600
    # length=600
)


g20200309_043555_50_lf = TensorInfo(
    name="lf",
    root="GKRESHUK",
    location="LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200309_Gcamp/fish1_awesome/2020-03-09_04.35.55/stack_3_channel_3/SinglePlane_-340/TP_*/RC_rectified/Cam_Right_*_rectified.tif",
    insert_singleton_axes_at=[0, 0],
    tag="g20200309_043555_50",
    # length=
)
g20200309_043555_50_ls = TensorInfo(
    name="ls",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20200309_Gcamp/fish1_awesome/2020-03-09_04.35.55/stack_3_channel_3/Cam_Left_*.h5/Data",
    insert_singleton_axes_at=[0, 0],
    z_slice=50,
    tag="g20200309_043555_50",
    samples_per_dataset=600,
    # length=600
)


g20200309_043555_40_lf = TensorInfo(
    name="lf",
    root="GKRESHUK",
    location="LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200309_Gcamp/fish1_awesome/2020-03-09_04.35.55/stack_4_channel_3/SinglePlane_-350/TP_*/RC_rectified/Cam_Right_*_rectified.tif",
    insert_singleton_axes_at=[0, 0],
    tag="g20200309_043555_40",
    # length=
)
g20200309_043555_40_ls = TensorInfo(
    name="ls",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20200309_Gcamp/fish1_awesome/2020-03-09_04.35.55/stack_4_channel_3/Cam_Left_*.h5/Data",
    insert_singleton_axes_at=[0, 0],
    z_slice=40,
    tag="g20200309_043555_40",
    samples_per_dataset=600,
    # length=600
)


g20200309_043555_30_lf = TensorInfo(
    name="lf",
    root="GKRESHUK",
    location="LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200309_Gcamp/fish1_awesome/2020-03-09_04.35.55/stack_5_channel_3/SinglePlane_-360/TP_*/RC_rectified/Cam_Right_*_rectified.tif",
    insert_singleton_axes_at=[0, 0],
    tag="g20200309_043555_30",
    # length=
)
g20200309_043555_30_ls = TensorInfo(
    name="ls",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20200309_Gcamp/fish1_awesome/2020-03-09_04.35.55/stack_5_channel_3/Cam_Left_*.h5/Data",
    insert_singleton_axes_at=[0, 0],
    z_slice=30,
    tag="g20200309_043555_30",
    samples_per_dataset=600,
    # length=600
)


g20200309_043555_20_lf = TensorInfo(
    name="lf",
    root="GKRESHUK",
    location="LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200309_Gcamp/fish1_awesome/2020-03-09_04.35.55/stack_6_channel_3/SinglePlane_-370/TP_*/RC_rectified/Cam_Right_*_rectified.tif",
    insert_singleton_axes_at=[0, 0],
    tag="g20200309_043555_20",
    # length=
)
g20200309_043555_20_ls = TensorInfo(
    name="ls",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20200309_Gcamp/fish1_awesome/2020-03-09_04.35.55/stack_6_channel_3/Cam_Left_*.h5/Data",
    insert_singleton_axes_at=[0, 0],
    z_slice=20,
    tag="g20200309_043555_20",
    samples_per_dataset=600,
    # length=600
)


g20200309_043555_70_lf = TensorInfo(
    name="lf",
    root="GKRESHUK",
    location="LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200309_Gcamp/fish1_awesome/2020-03-09_04.35.55/stack_7_channel_3/SinglePlane_-320/TP_*/RC_rectified/Cam_Right_*_rectified.tif",
    insert_singleton_axes_at=[0, 0],
    tag="g20200309_043555_70",
    # length=
)
g20200309_043555_70_ls = TensorInfo(
    name="ls",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20200309_Gcamp/fish1_awesome/2020-03-09_04.35.55/stack_7_channel_3/Cam_Left_*.h5/Data",
    insert_singleton_axes_at=[0, 0],
    z_slice=70,
    tag="g20200309_043555_70",
    samples_per_dataset=600,
    # length=600
)


g20200309_043555_80_lf = TensorInfo(
    name="lf",
    root="GKRESHUK",
    location="LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200309_Gcamp/fish1_awesome/2020-03-09_04.35.55/stack_8_channel_3/SinglePlane_-310/TP_*/RC_rectified/Cam_Right_*_rectified.tif",
    insert_singleton_axes_at=[0, 0],
    tag="g20200309_043555_80",
    # length=
)
g20200309_043555_80_ls = TensorInfo(
    name="ls",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20200309_Gcamp/fish1_awesome/2020-03-09_04.35.55/stack_8_channel_3/Cam_Left_*.h5/Data",
    insert_singleton_axes_at=[0, 0],
    z_slice=80,
    tag="g20200309_043555_80",
    samples_per_dataset=600,
    # length=600
)


g20200309_043555_90_lf = TensorInfo(
    name="lf",
    root="GKRESHUK",
    location="LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200309_Gcamp/fish1_awesome/2020-03-09_04.35.55/stack_9_channel_3/SinglePlane_-300/TP_*/RC_rectified/Cam_Right_*_rectified.tif",
    insert_singleton_axes_at=[0, 0],
    tag="g20200309_043555_90",
    # length=
)
g20200309_043555_90_ls = TensorInfo(
    name="ls",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20200309_Gcamp/fish1_awesome/2020-03-09_04.35.55/stack_9_channel_3/Cam_Left_*.h5/Data",
    insert_singleton_axes_at=[0, 0],
    z_slice=90,
    tag="g20200309_043555_90",
    samples_per_dataset=600,
    # length=600
)


g20200309_043555_100_lf = TensorInfo(
    name="lf",
    root="GKRESHUK",
    location="LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200309_Gcamp/fish1_awesome/2020-03-09_04.35.55/stack_10_channel_3/SinglePlane_-290/TP_*/RC_rectified/Cam_Right_*_rectified.tif",
    insert_singleton_axes_at=[0, 0],
    tag="g20200309_043555_100",
    # length=
)
g20200309_043555_100_ls = TensorInfo(
    name="ls",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20200309_Gcamp/fish1_awesome/2020-03-09_04.35.55/stack_10_channel_3/Cam_Left_*.h5/Data",
    insert_singleton_axes_at=[0, 0],
    z_slice=100,
    tag="g20200309_043555_100",
    samples_per_dataset=600,
    # length=600
)


g20200309_043555_0_lf = TensorInfo(
    name="lf",
    root="GKRESHUK",
    location="LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200309_Gcamp/fish1_awesome/2020-03-09_04.35.55/stack_12_channel_3/SinglePlane_-390/TP_*/RC_rectified/Cam_Right_*_rectified.tif",
    insert_singleton_axes_at=[0, 0],
    tag="g20200309_043555_0",
    # length=
)
g20200309_043555_0_ls = TensorInfo(
    name="ls",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20200309_Gcamp/fish1_awesome/2020-03-09_04.35.55/stack_12_channel_3/Cam_Left_*.h5/Data",
    insert_singleton_axes_at=[0, 0],
    z_slice=0,
    tag="g20200309_043555_0",
    samples_per_dataset=600,
    # length=600
)


g20200309_043555_10_lf = TensorInfo(
    name="lf",
    root="GKRESHUK",
    location="LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200309_Gcamp/fish1_awesome/2020-03-09_04.35.55/stack_13_channel_3/SinglePlane_-380/TP_*/RC_rectified/Cam_Right_*_rectified.tif",
    insert_singleton_axes_at=[0, 0],
    tag="g20200309_043555_10",
    # length=
)
g20200309_043555_10_ls = TensorInfo(
    name="ls",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20200309_Gcamp/fish1_awesome/2020-03-09_04.35.55/stack_13_channel_3/Cam_Left_*.h5/Data",
    insert_singleton_axes_at=[0, 0],
    z_slice=10,
    tag="g20200309_043555_10",
    samples_per_dataset=600,
    # length=600
)


g20200309_043555_110_lf = TensorInfo(
    name="lf",
    root="GKRESHUK",
    location="LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200309_Gcamp/fish1_awesome/2020-03-09_04.35.55/stack_14_channel_3/SinglePlane_-280/TP_*/RC_rectified/Cam_Right_*_rectified.tif",
    insert_singleton_axes_at=[0, 0],
    tag="g20200309_043555_110",
    # length=
)
g20200309_043555_110_ls = TensorInfo(
    name="ls",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20200309_Gcamp/fish1_awesome/2020-03-09_04.35.55/stack_14_channel_3/Cam_Left_*.h5/Data",
    insert_singleton_axes_at=[0, 0],
    z_slice=110,
    tag="g20200309_043555_110",
    samples_per_dataset=600,
    # length=600
)


g20200309_043555_120_lf = TensorInfo(
    name="lf",
    root="GKRESHUK",
    location="LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200309_Gcamp/fish1_awesome/2020-03-09_04.35.55/stack_15_channel_3/SinglePlane_-270/TP_*/RC_rectified/Cam_Right_*_rectified.tif",
    insert_singleton_axes_at=[0, 0],
    tag="g20200309_043555_120",
    # length=
)
g20200309_043555_120_ls = TensorInfo(
    name="ls",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20200309_Gcamp/fish1_awesome/2020-03-09_04.35.55/stack_15_channel_3/Cam_Left_*.h5/Data",
    insert_singleton_axes_at=[0, 0],
    z_slice=120,
    tag="g20200309_043555_120",
    samples_per_dataset=600,
    # length=600
)


# single plane -330
# = TensorInfo(
#     name="lf",
#     root="GKRESHUK",
#     location="LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish1/5Hz/longRun/niceOne/2020-03-11_03.43.42/stack_34_channel_4",
#     insert_singleton_axes_at=[0, 0],
# )
# = TensorInfo(
#     name="lf",
#     root="GKRESHUK",
#     location="LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish1/5Hz/longRun/niceOne3/2020-03-11_04.25.22",
#     insert_singleton_axes_at=[0, 0],
# )
# = TensorInfo(
#     name="lf",
#     root="GKRESHUK",
#     location="LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish2/5Hz/2020-03-11_06.53.14/stack_34_channel_4/SinglePlane_-330",
#     insert_singleton_axes_at=[0, 0],
# )

if __name__ == "__main__":
    import logging

    from hashlib import sha224 as hash_algorithm
    from lnet.datasets.base import get_dataset_from_info, N5CachedDatasetFromInfo

    logger = logging.getLogger("gcamp")

    for info_lf, info_ls in zip(
        [
            # g200311_085233_lf,
            # g200311_090800_lf,
            # g200311_083021_lf,
            # g200311_083419_lf,
            # g200311_083658_lf,
            # g200311_083931_lf,
            # g200311_084207_lf,
            # g200311_084450_lf,
            # g200311_084729_lf,
            # g20200311_073039_70_lf,
            # g20200311_073039_80_lf,
            # g20200311_101320_100_lf,
            # g20200311_101320_75_lf,
            # g20200311_102541_50_lf,
            g20200311_102541_26_lf,
            # g20200311_102541_85_lf,
            # g20200311_065726_lf,
            # g20200311_073447_lf,
            # g20200311_033035_lf,
            # g20200311_040327_lf,
            # g20200311_042734_lf,
            # g20200311_042839_lf,
            # g20200311_043502_lf,
            # g20200311_024927_lf,
            # g20200311_101734_110_lf,
            # g20200311_101734_60_lf,
            g20200311_102114_26_lf,
            # g20200311_102114_85_lf,
            # g20200309_043555_60_lf,
            # g20200309_043555_50_lf,
            # g20200309_043555_40_lf,
            # g20200309_043555_30_lf,
            # g20200309_043555_20_lf,
            # g20200309_043555_70_lf,
            # g20200309_043555_80_lf,
            # g20200309_043555_90_lf,
            # g20200309_043555_100_lf,
            # g20200309_043555_0_lf,
            # g20200309_043555_10_lf,
            # g20200309_043555_110_lf,
            # g20200309_043555_120_lf,
        ],
        [
            # g200311_085233_ls,
            # g200311_090800_ls,
            # g200311_083021_ls,
            # g200311_083419_ls,
            # g200311_083658_ls,
            # g200311_083931_ls,
            # g200311_084207_ls,
            # g200311_084450_ls,
            # g200311_084729_ls,
            # g20200311_073039_70_ls,
            # g20200311_073039_80_ls,
            # g20200311_101320_100_ls,
            # g20200311_101320_75_ls,
            # g20200311_102541_50_ls,
            g20200311_102541_26_ls,
            # g20200311_102541_85_ls,
            # g20200311_065726_ls,
            # g20200311_073447_ls,
            # g20200311_033035_ls,
            # g20200311_040327_ls,
            # g20200311_042734_ls,
            # g20200311_042839_ls,
            # g20200311_043502_ls,
            # g20200311_024927_ls,
            # g20200311_101734_110_ls,
            # g20200311_101734_60_ls,
            g20200311_102114_26_ls,
            # g20200311_102114_85_ls,
            # g20200309_043555_60_ls,
            # g20200309_043555_50_ls,
            # g20200309_043555_40_ls,
            # g20200309_043555_30_ls,
            # g20200309_043555_20_ls,
            # g20200309_043555_70_ls,
            # g20200309_043555_80_ls,
            # g20200309_043555_90_ls,
            # g20200309_043555_100_ls,
            # g20200309_043555_0_ls,
            # g20200309_043555_10_ls,
            # g20200309_043555_110_ls,
            # g20200309_043555_120_ls,
        ],
    ):
        # try:
        info_lf.transformations += [
            {"Assert": {"apply_to": "lf", "expected_tensor_shape": [None, 1, None, None]}}
        ]  # bcyx
        dslf = get_dataset_from_info(info_lf)
        print(
            settings.cache_path
            / f"{dslf.info.tag}_{dslf.tensor_name}_{hash_algorithm(dslf.description.encode()).hexdigest()}.n5"
        )
        dslf = N5CachedDatasetFromInfo(dslf)

        print(len(dslf))
        print(dslf[0]["lf"].shape)
        info_ls.transformations += [
            {
                "Resize": {
                    "apply_to": "ls",
                    # "shape": [1.0, 1.0, 0.21052631578947368421052631578947, 0.21052631578947368421052631578947],
                    "shape": [1.0, 1.0, 0.42105263157894736842105263157895, 0.42105263157894736842105263157895],
                    "order": 2,
                }
            },
            {"Assert": {"apply_to": "ls", "expected_tensor_shape": [None, 1, 1, None, None]}},
        ]
        dsls = get_dataset_from_info(info_ls)
        print(
            settings.cache_path
            / f"{dsls.info.tag}_{dsls.tensor_name}_{hash_algorithm(dsls.description.encode()).hexdigest()}.n5"
        )
        dsls = N5CachedDatasetFromInfo(dsls)

        print(len(dsls))
        print(dsls[0]["ls"].shape)
        assert len(dslf) == len(dsls), (len(dslf), len(dsls))
        # except Exception as e:
        #     print("error")
        #     logger.error(e, exc_info=True)
