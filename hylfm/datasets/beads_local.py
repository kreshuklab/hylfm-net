from hylfm.datasets.base import TensorInfo

b01highc_2_ls = TensorInfo(
    name="ls_reg",
    root="GHUFNAGELLFLenseLeNet_Microscope",
    location="20191031_Beads_MixedSizes/Beads_01micron_highConcentration/2019-10-31_04.57.13/stack_1_channel_1/TP_*/LC/Cam_Left_registered.tif",
    insert_singleton_axes_at=[0, 0],
    tag="b01highc_2",
    # length=137,
)