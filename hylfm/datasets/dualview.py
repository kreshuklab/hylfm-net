from hylfm.datasets import TensorInfo

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


def get_tensor_info(tag: str, name: str, meta: dict):
    root = "GKRESHUK"
    if tag == "RC_LFD_n156to156_steps4":
        if name == "lfd":
            location = "LF_computed/LenseLeNet_Microscope/dualview_060918_added/RC_LFD_-156to156_steps4/Cam_Right_*.tif"
        elif name == "lf":
            location = "LF_computed/LenseLeNet_Microscope/dualview_060918_added/RC_rectified/Cam_Right_*.tif"
        else:
            raise NotImplementedError(tag, name)
    elif tag == "LC_LFD_n156to156_steps4":
        if name == "lfd":
            location = "LF_computed/LenseLeNet_Microscope/dualview_060918_added/LC_LFD_-156to156_steps4/Cam_Right_*.tif"
        elif name == "lf":
            location = "LF_computed/LenseLeNet_Microscope/dualview_060918_added/LC_rectified/Cam_Right_*.tif"
        else:
            raise NotImplementedError(tag, name)

    else:
        raise NotImplementedError(tag, name)

    return TensorInfo(name=name, root=root, location=location, insert_singleton_axes_at=[0, 0], tag=tag)
