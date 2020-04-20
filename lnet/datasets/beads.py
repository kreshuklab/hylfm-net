from lnet.datasets.base import TensorInfo

b01mu_0_lf = TensorInfo(
    name="lf",
    root="GHUFNAGELLFLenseLeNet_Microscope",
    location="20191030_Beads_massiveGT/Beads_1micron/2019-10-30_05.20.28_withSimultIllum/stack_0_channel_0/TP_*/RC_rectified/Cam_Right_1_rectified.tif",
    skip_indices=[0],  # registered ls has shape (828, 931, 1406) whereas all others have shape (838, 931, 1406)
    insert_singleton_axes_at=[0, 0],
)
b01mu_0_ls = TensorInfo(
    name="ls",
    root="GHUFNAGELLFLenseLeNet_Microscope",
    location="20191030_Beads_massiveGT/Beads_1micron/2019-10-30_05.20.28_withSimultIllum/stack_1_channel_1/TP_*/LC/Cam_Left_registered.tif",
    skip_indices=[0],  # registered ls has shape (828, 931, 1406) whereas all others have shape (838, 931, 1406)
    insert_singleton_axes_at=[0, 0],
)

b01mu_0_lr = TensorInfo(
    name="lr",
    root="GHUFNAGELLFLenseLeNet_Microscope",
    location="20191030_Beads_massiveGT/Beads_1micron/2019-10-30_05.20.28_withSimultIllum/stack_0_channel_0/TP_*/RCout/Cam_Right_1.tif",
    skip_indices=[0],  # registered ls has shape (828, 931, 1406) whereas all others have shape (838, 931, 1406)
    insert_singleton_axes_at=[0, 0],
)


b01highc_0_lf = TensorInfo(
    name="lf",
    root="GHUFNAGELLFLenseLeNet_Microscope",
    location="20191031_Beads_MixedSizes/Beads_01micron_highConcentration/2019-10-31_02.57.02/stack_0_channel_0/TP_*/RC_rectified/Cam_Right_1_rectified.tif",
    insert_singleton_axes_at=[0, 0],
    # length=4,
)

b01highc_0_ls = TensorInfo(
    name="ls",
    root="GHUFNAGELLFLenseLeNet_Microscope",
    location="20191031_Beads_MixedSizes/Beads_01micron_highConcentration/2019-10-31_02.57.02/stack_1_channel_1/TP_*/LC/Cam_Left_registered.tif",
    insert_singleton_axes_at=[0, 0],
    # length=4,
)

b01highc_0_lr = TensorInfo(
    name="lr",
    root="GHUFNAGELLFLenseLeNet_Microscope",
    location="20191031_Beads_MixedSizes/Beads_01micron_highConcentration/2019-10-31_02.57.02/stack_0_channel_0/TP_*/RCout/Cam_Right_1.tif",
    insert_singleton_axes_at=[0, 0],
    # length=4,
)


b01highc_1_lf = TensorInfo(
    name="lf",
    root="GHUFNAGELLFLenseLeNet_Microscope",
    location="20191031_Beads_MixedSizes/Beads_01micron_highConcentration/2019-10-31_03.01.49/stack_0_channel_0/TP_*/RC_rectified/Cam_Right_1_rectified.tif",
    insert_singleton_axes_at=[0, 0],
    # length=28,
)

b01highc_1_ls = TensorInfo(
    name="ls",
    root="GHUFNAGELLFLenseLeNet_Microscope",
    location="20191031_Beads_MixedSizes/Beads_01micron_highConcentration/2019-10-31_03.01.49/stack_1_channel_1/TP_*/LC/Cam_Left_registered.tif",
    insert_singleton_axes_at=[0, 0],
    # length=28,
)

b01highc_1_lr = TensorInfo(
    name="lr",
    root="GHUFNAGELLFLenseLeNet_Microscope",
    location="20191031_Beads_MixedSizes/Beads_01micron_highConcentration/2019-10-31_03.01.49/stack_0_channel_0/TP_*/RCout/Cam_Right_1.tif",
    insert_singleton_axes_at=[0, 0],
    # length=28,
)


b01highc_2_lf = TensorInfo(
    name="lf",
    root="GHUFNAGELLFLenseLeNet_Microscope",
    location="20191031_Beads_MixedSizes/Beads_01micron_highConcentration/2019-10-31_04.57.13/stack_0_channel_0/TP_*/RC_rectified/Cam_Right_1_rectified.tif",
    insert_singleton_axes_at=[0, 0],
    # length=137,
)

b01highc_2_ls = TensorInfo(
    name="ls",
    root="GHUFNAGELLFLenseLeNet_Microscope",
    location="20191031_Beads_MixedSizes/Beads_01micron_highConcentration/2019-10-31_04.57.13/stack_1_channel_1/TP_*/LC/Cam_Left_registered.tif",
    insert_singleton_axes_at=[0, 0],
    # length=137,
)

b01highc_2_lr = TensorInfo(
    name="lr",
    root="GHUFNAGELLFLenseLeNet_Microscope",
    location="20191031_Beads_MixedSizes/Beads_01micron_highConcentration/2019-10-31_04.57.13/stack_0_channel_0/TP_*/RCout/Cam_Right_1.tif",
    insert_singleton_axes_at=[0, 0],
    # length=137,
)


if __name__ == "__main__":
    from lnet.datasets.base import get_dataset_from_info, N5CachedDatasetFromInfo

    ds = get_dataset_from_info(b01mu_0_lf)
    ds = N5CachedDatasetFromInfo(ds)
    print(len(ds))
    print(ds[0]["lf"].shape)


# beads_01mu_1 = TensorInfo(
#
#         "/g/hufnagel/LF/LenseLeNet_Microscope/20191030_Beads_massiveGT/Beads_1micron/2019-10-30_05.54.12_withSimultIllum"
#     ),
#     "stack_0_channel_0/TP_*/RC_rectified/Cam_Right_1_rectified.tif",
#     "stack_1_channel_1/TP_*/LC/Cam_Left_registered.tif",
#     description="beads_01mu_1",
# )
#
# beads_01mu_2 = TensorInfo(
# GHUFNAGELLFLenseLeNet_Microscope) / "/20191030_Beads_massiveGT/Beads_1micron/2019-10-30_06.25.24",
#     "stack_0_channel_0/TP_*/RC_rectified/Cam_Right_1_rectified.tif",
#     "stack_1_channel_1/TP_*/LC/Cam_Left_registered.tif",
#     description="beads_1mu_2",
# )
#
# beads_01mu_3 = TensorInfo(
# GHUFNAGELLFLenseLeNet_Microscope) / "/20191030_Beads_massiveGT/Beads_1micron/2019-10-30_06.44.56",
#     "stack_0_channel_0/TP_*/RC_rectified/Cam_Right_1_rectified.tif",
#     "stack_1_channel_1/TP_*/LC/Cam_Left_registered.tif",
#     description="beads_1mu_3",
# )
#
# beads_01mu_4 = TensorInfo(
# GHUFNAGELLFLenseLeNet_Microscope) / "/20191030_Beads_massiveGT/Beads_1micron/2019-10-30_07.04.52",
#     "stack_0_channel_0/TP_*/RC_rectified/Cam_Right_1_rectified.tif",
#     "stack_1_channel_1/TP_*/LC/Cam_Left_registered.tif",
#     description="beads_1mu_4",
# )
#
# beads_01mu_5 = TensorInfo(
# GHUFNAGELLFLenseLeNet_Microscope) / "/20191030_Beads_massiveGT/Beads_1micron/2019-10-30_07.23.55",
#     "stack_0_channel_0/TP_*/RC_rectified/Cam_Right_1_rectified.tif",
#     "stack_1_channel_1/TP_*/LC/Cam_Left_registered.tif",
#     description="beads_1mu_5",
# )
