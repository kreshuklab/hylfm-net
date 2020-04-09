from .base import TensorInfo

ref0_lf = TensorInfo(
    "lf", "lnet", "ref_data/AffineTransforms/SwipeThrough_-450_-210_nimages_241/*Cam_Right.tif", transforms=[], meta={}, insert_singleton_axes_at=[0, 0]
)
ref0_sample_ls_slice = TensorInfo(
    "ls_slice", "lnet", "ref_data/AffineTransforms/SwipeThrough_-450_-210_nimages_241/*Cam_Left.tif", transforms=[], meta={}, insert_singleton_axes_at=[0, 0]
)

ref0_lr = TensorInfo(
    "lr", "lnet", "ref_data/AffineTransforms/SwipeThrough_-450_-210_nimages_241/Gcamp_dataset.h5/t[0-9]+/s00/0/cells", transforms=[], meta={}, insert_singleton_axes_at=[0, 0]
)
ref0_ls = TensorInfo(
    "ls", "lnet", "ref_data/AffineTransforms/SwipeThrough_-450_-210_nimages_241/Gcamp_dataset.h5/t[0-9]+/s01/0/cells", transforms=[], meta={}, insert_singleton_axes_at=[0, 0]
)


# gcamp_0311_085233 = NamedDatasetInfo(
#     Path(GKRESHUK),
#     "LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish2/10Hz/241Planes/2020-03-11_08.52.33/stack_1_channel_3/SwipeThrough_-450_-210_nimages_241/TP_*/RC_rectified/Cam_Right_*_rectified.tif",
#     "LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish2/10Hz/241Planes/2020-03-11_08.52.33/stack_1_channel_3/SwipeThrough_-450_-210_nimages_241/TP_*/LC/Cam_Left_*.tif",
#     description="gcamp 2020/03/11 08:52:33",
#     # length=,
#     # x_shape=(,),
#     # y_shape=(,,),
# )
