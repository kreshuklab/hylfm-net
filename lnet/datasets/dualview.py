from lnet.datasets import TensorInfo

beads_right = TensorInfo(
    name="lf",
    root="GKRESHUK",
    location="LF_computed/LenseLeNet_Microscope/DualView_comparison_heart_movie/beads/2018-09-06_13.16.46/stack_2_channel_0/RC_rectified/*.tif",
    insert_singleton_axes_at=[0, 0],
    tag="dualbeads_right",
)

beads_left = TensorInfo(
    name="lf",
    root="GKRESHUK",
    location="LF_computed/LenseLeNet_Microscope/DualView_comparison_heart_movie/beads/2018-09-06_13.16.46/stack_2_channel_0/LC_rectified/*.tif",
    insert_singleton_axes_at=[0, 0],
    tag="dualbeads_left",
)


heart_right = TensorInfo(
    name="lf",
    root="GKRESHUK",
    location="LF_computed/LenseLeNet_Microscope/DualView_comparison_heart_movie/heart/Rectified_RC/*.tif",
    insert_singleton_axes_at=[0, 0],
    tag="dualheart_right",
)

heart_left = TensorInfo(
    name="lf",
    root="GKRESHUK",
    location="LF_computed/LenseLeNet_Microscope/DualView_comparison_heart_movie/heart/Rectified_LC/*.tif",
    insert_singleton_axes_at=[0, 0],
    tag="dualheart_left",
)
