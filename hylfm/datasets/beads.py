from hylfm import settings
from hylfm.datasets.base import TensorInfo
from hylfm.datasets.online import OnlineTensorInfo


def get_tensor_info(tag: str, name: str, meta: dict):
    if tag == "small_0":
        doi = "10.5281/zenodo.4019246"
        archive_name_with_suffix = f"{tag}_{name}.zip"
        glob_expression = "TP_*.tif"
    elif tag == "small_1":
        doi = "10.5281/zenodo.4020352"
        archive_name_with_suffix = f"{tag}_{name}.zip"
        glob_expression = "TP_*.tif"
    else:
        raise NotImplementedError(tag, name)

    info = OnlineTensorInfo(
        name=name,
        doi=doi,
        archive_name_with_suffix=archive_name_with_suffix,
        glob_expression=glob_expression,
        meta=meta,
        insert_singleton_axes_at=[0, 0],
        tag=f"{tag}_{name}",
    )
    info.download()
    info.extract()
    return info


b01highc_0_lf = TensorInfo(
    name="lf",
    root="GHUFNAGELLFLenseLeNet_Microscope",
    location="20191031_Beads_MixedSizes/Beads_01micron_highConcentration/2019-10-31_02.57.02/stack_0_channel_0/TP_*/RC_rectified/Cam_Right_1_rectified.tif",
    insert_singleton_axes_at=[0, 0],
    tag="b01highc_0"
    # length=4,
)

b01highc_0_ls = TensorInfo(
    name="ls_reg",
    root="GHUFNAGELLFLenseLeNet_Microscope",
    location="20191031_Beads_MixedSizes/Beads_01micron_highConcentration/2019-10-31_02.57.02/stack_1_channel_1/TP_*/LC/Cam_Left_registered.tif",
    insert_singleton_axes_at=[0, 0],
    tag="b01highc_0"
    # length=4,
)

b01highc_0_lr = TensorInfo(
    name="lr",
    root="GHUFNAGELLFLenseLeNet_Microscope",
    location="20191031_Beads_MixedSizes/Beads_01micron_highConcentration/2019-10-31_02.57.02/stack_0_channel_0/TP_*/RCout/Cam_Right_1.tif",
    insert_singleton_axes_at=[0, 0],
    tag="b01highc_0"
    # length=4,
)


b01highc_1_lf = TensorInfo(
    name="lf",
    root="GHUFNAGELLFLenseLeNet_Microscope",
    location="20191031_Beads_MixedSizes/Beads_01micron_highConcentration/2019-10-31_03.01.49/stack_0_channel_0/TP_*/RC_rectified/Cam_Right_1_rectified.tif",
    insert_singleton_axes_at=[0, 0],
    tag="b01highc_1"
    # length=28,
)

b01highc_1_ls = TensorInfo(
    name="ls_reg",
    root="GHUFNAGELLFLenseLeNet_Microscope",
    location="20191031_Beads_MixedSizes/Beads_01micron_highConcentration/2019-10-31_03.01.49/stack_1_channel_1/TP_*/LC/Cam_Left_registered.tif",
    insert_singleton_axes_at=[0, 0],
    tag="b01highc_1"
    # length=28,
)

b01highc_1_lr = TensorInfo(
    name="lr",
    root="GHUFNAGELLFLenseLeNet_Microscope",
    location="20191031_Beads_MixedSizes/Beads_01micron_highConcentration/2019-10-31_03.01.49/stack_0_channel_0/TP_*/RCout/Cam_Right_1.tif",
    insert_singleton_axes_at=[0, 0],
    tag="b01highc_1"
    # length=28,
)

b01highc_1_lr_z51 = TensorInfo(
    name="lr",
    root="GHUFNAGELLFLenseLeNet_Microscope",
    location="20191031_Beads_MixedSizes/Beads_01micron_highConcentration/2019-10-31_03.01.49/stack_0_channel_0/TP_*/RCout_51planes/Cam_Right_1.tif",
    insert_singleton_axes_at=[0, 0],
    tag="b01highc_1_z51"
    # length=28,
)


b01highc_2_lf = TensorInfo(
    name="lf",
    root="GHUFNAGELLFLenseLeNet_Microscope",
    location="20191031_Beads_MixedSizes/Beads_01micron_highConcentration/2019-10-31_04.57.13/stack_0_channel_0/TP_*/RC_rectified/Cam_Right_1_rectified.tif",
    insert_singleton_axes_at=[0, 0],
    tag="b01highc_2",
    # length=137,
)

b01highc_2_ls = TensorInfo(
    name="ls_reg",
    root="GHUFNAGELLFLenseLeNet_Microscope",
    location="20191031_Beads_MixedSizes/Beads_01micron_highConcentration/2019-10-31_04.57.13/stack_1_channel_1/TP_*/LC/Cam_Left_registered.tif",
    insert_singleton_axes_at=[0, 0],
    tag="b01highc_2",
    # length=137,
)

b01highc_2_lr = TensorInfo(
    name="lr",
    root="GHUFNAGELLFLenseLeNet_Microscope",
    location="20191031_Beads_MixedSizes/Beads_01micron_highConcentration/2019-10-31_04.57.13/stack_0_channel_0/TP_*/RCout/Cam_Right_1.tif",
    insert_singleton_axes_at=[0, 0],
    tag="b01highc_2",
    # length=137,
)

# 11mu
b11mu_0_lf = TensorInfo(
    name="lf",
    root="GHUFNAGELLFLenseLeNet_Microscope",
    location="20191030_Beads_massiveGT/Beads_11micron/2019-10-30_09.37.45/stack_0_channel_0/TP_*/RC_rectified/Cam_Right_1_rectified.tif",
    tag="b11mu_0",
)

b11mu_0_lr = TensorInfo(
    name="lr",
    root="GHUFNAGELLFLenseLeNet_Microscope",
    location="20191030_Beads_massiveGT/Beads_11micron/2019-10-30_09.37.45/stack_0_channel_0/TP_*/RCout/Cam_Right_1.tif",
    tag="b11mu_0",
)

b11mu_0_ls = TensorInfo(
    name="ls_reg",
    root="GHUFNAGELLFLenseLeNet_Microscope",
    location="20191030_Beads_massiveGT/Beads_11micron/2019-10-30_09.37.45/stack_1_channel_1/TP_*/LC/Cam_Left_registered.tif",
    tag="b11mu_0",
)


# 0.1mu
b01mu_0_lf = TensorInfo(
    name="lf",
    root="GHUFNAGELLFLenseLeNet_Microscope",
    location="20191030_Beads_massiveGT/Beads_1micron/2019-10-30_05.20.28_withSimultIllu/stack_0_channel_0/TP_*/RC_rectified/Cam_Right_1_rectified.tif",
    tag="b01mu_0",
    skip_indices=[0],  # registered ls has shape (828, 931, 1406) whereas all others have shape (838, 931, 1406)
    insert_singleton_axes_at=[0, 0],
)

b01mu_0_lr = TensorInfo(
    name="lr",
    root="GHUFNAGELLFLenseLeNet_Microscope",
    location="20191030_Beads_massiveGT/Beads_1micron/2019-10-30_05.20.28_withSimultIllu/stack_0_channel_0/TP_*/RCout/Cam_Right_1.tif",
    tag="b01mu_0",
    skip_indices=[0],  # registered ls has shape (828, 931, 1406) whereas all others have shape (838, 931, 1406)
    insert_singleton_axes_at=[0, 0],
)

b01mu_0_ls = TensorInfo(
    name="ls_reg",
    root="GHUFNAGELLFLenseLeNet_Microscope",
    location="20191030_Beads_massiveGT/Beads_1micron/2019-10-30_05.20.28_withSimultIllu/stack_1_channel_1/TP_*/LC/Cam_Left_registered.tif",
    tag="b01mu_0",
    skip_indices=[0],  # registered ls has shape (828, 931, 1406) whereas all others have shape (838, 931, 1406)
    insert_singleton_axes_at=[0, 0],
)

b01mu_1_lf = TensorInfo(
    name="lf",
    root="GHUFNAGELLFLenseLeNet_Microscope",
    location="20191030_Beads_massiveGT/Beads_1micron/2019-10-30_05.54.12_withSimultIllu/stack_0_channel_0/TP_*/RC_rectified/Cam_Right_1_rectified.tif",
    tag="b01mu_1",
    insert_singleton_axes_at=[0, 0],
)

b01mu_1_lr = TensorInfo(
    name="lr",
    root="GHUFNAGELLFLenseLeNet_Microscope",
    location="20191030_Beads_massiveGT/Beads_1micron/2019-10-30_05.54.12_withSimultIllu/stack_0_channel_0/TP_*/RCout/Cam_Right_1.tif",
    tag="b01mu_1",
    insert_singleton_axes_at=[0, 0],
)

b01mu_1_ls = TensorInfo(
    name="ls_reg",
    root="GHUFNAGELLFLenseLeNet_Microscope",
    location="20191030_Beads_massiveGT/Beads_1micron/2019-10-30_05.54.12_withSimultIllu/stack_1_channel_1/TP_*/LC/Cam_Left_registered.tif",
    tag="b01mu_1",
    insert_singleton_axes_at=[0, 0],
)

b01mu_2_lf = TensorInfo(
    name="lf",
    root="GHUFNAGELLFLenseLeNet_Microscope",
    location="20191030_Beads_massiveGT/Beads_1micron/2019-10-30_06.25.24/stack_0_channel_0/TP_*/RC_rectified/Cam_Right_1_rectified.tif",
    tag="b1mu_2",
    insert_singleton_axes_at=[0, 0],
)

b01mu_2_lr = TensorInfo(
    name="lr",
    root="GHUFNAGELLFLenseLeNet_Microscope",
    location="20191030_Beads_massiveGT/Beads_1micron/2019-10-30_06.25.24/stack_0_channel_0/TP_*/RCout/Cam_Right_1.tif",
    tag="b1mu_2",
    insert_singleton_axes_at=[0, 0],
)

b01mu_2_ls = TensorInfo(
    name="ls_reg",
    root="GHUFNAGELLFLenseLeNet_Microscope",
    location="20191030_Beads_massiveGT/Beads_1micron/2019-10-30_06.25.24/stack_1_channel_1/TP_*/LC/Cam_Left_registered.tif",
    tag="b1mu_2",
    insert_singleton_axes_at=[0, 0],
)

b01mu_3_lf = TensorInfo(
    name="lf",
    root="GHUFNAGELLFLenseLeNet_Microscope",
    location="20191030_Beads_massiveGT/Beads_1micron/2019-10-30_06.44.56/stack_0_channel_0/TP_*/RC_rectified/Cam_Right_1_rectified.tif",
    tag="b1mu_3",
    insert_singleton_axes_at=[0, 0],
)

b01mu_3_lr = TensorInfo(
    name="lr",
    root="GHUFNAGELLFLenseLeNet_Microscope",
    location="20191030_Beads_massiveGT/Beads_1micron/2019-10-30_06.44.56/stack_0_channel_0/TP_*/RCout/Cam_Right_1.tif",
    tag="b1mu_3",
    insert_singleton_axes_at=[0, 0],
)

b01mu_3_ls = TensorInfo(
    name="ls_reg",
    root="GHUFNAGELLFLenseLeNet_Microscope",
    location="20191030_Beads_massiveGT/Beads_1micron/2019-10-30_06.44.56/stack_1_channel_1/TP_*/LC/Cam_Left_registered.tif",
    tag="b1mu_3",
    insert_singleton_axes_at=[0, 0],
)

b01mu_4_lf = TensorInfo(
    name="lf",
    root="GHUFNAGELLFLenseLeNet_Microscope",
    location="20191030_Beads_massiveGT/Beads_1micron/2019-10-30_07.04.52/stack_0_channel_0/TP_*/RC_rectified/Cam_Right_1_rectified.tif",
    tag="b1mu_4",
    insert_singleton_axes_at=[0, 0],
)

b01mu_4_lr = TensorInfo(
    name="lr",
    root="GHUFNAGELLFLenseLeNet_Microscope",
    location="20191030_Beads_massiveGT/Beads_1micron/2019-10-30_07.04.52/stack_0_channel_0/TP_*/RCout/Cam_Right_1.tif",
    tag="b1mu_4",
    insert_singleton_axes_at=[0, 0],
)

b01mu_4_ls = TensorInfo(
    name="ls_reg",
    root="GHUFNAGELLFLenseLeNet_Microscope",
    location="20191030_Beads_massiveGT/Beads_1micron/2019-10-30_07.04.52/stack_1_channel_1/TP_*/LC/Cam_Left_registered.tif",
    tag="b1mu_4",
    insert_singleton_axes_at=[0, 0],
)

b01mu_5_lf = TensorInfo(
    name="lf",
    root="GHUFNAGELLFLenseLeNet_Microscope",
    location="20191030_Beads_massiveGT/Beads_1micron/2019-10-30_07.23.55/stack_0_channel_0/TP_*/RC_rectified/Cam_Right_1_rectified.tif",
    tag="b1mu_5",
    insert_singleton_axes_at=[0, 0],
)

b01mu_5_lr = TensorInfo(
    name="lr",
    root="GHUFNAGELLFLenseLeNet_Microscope",
    location="20191030_Beads_massiveGT/Beads_1micron/2019-10-30_07.23.55/stack_0_channel_0/TP_*/RCout/Cam_Right_1.tif",
    tag="b1mu_5",
    insert_singleton_axes_at=[0, 0],
)

b01mu_5_ls = TensorInfo(
    name="ls_reg",
    root="GHUFNAGELLFLenseLeNet_Microscope",
    location="20191030_Beads_massiveGT/Beads_1micron/2019-10-30_07.23.55/stack_1_channel_1/TP_*/LC/Cam_Left_registered.tif",
    tag="b1mu_5",
    insert_singleton_axes_at=[0, 0],
)

b4mu_0_lf = TensorInfo(
    name="lf",
    root="GHUFNAGELLFLenseLeNet_Microscope",
    location="20191030_Beads_massiveGT/Beads_4micron/2019-10-30_08.15.32/stack_0_channel_0/TP_*/RC_rectified/Cam_Right_1_rectified.tif",
    tag="b4mu_0",
    insert_singleton_axes_at=[0, 0],
)

b4mu_0_lr = TensorInfo(
    name="lr",
    root="GHUFNAGELLFLenseLeNet_Microscope",
    location="20191030_Beads_massiveGT/Beads_4micron/2019-10-30_08.15.32/stack_0_channel_0/TP_*/RCout/Cam_Right_1.tif",
    tag="b4mu_0",
    insert_singleton_axes_at=[0, 0],
    # length=40,
)

b4mu_0_ls = TensorInfo(
    name="ls_reg",
    root="GHUFNAGELLFLenseLeNet_Microscope",
    location="20191030_Beads_massiveGT/Beads_4micron/2019-10-30_08.15.32/stack_1_channel_1/TP_*/LC/Cam_Left_registered.tif",
    tag="b4mu_0",
    insert_singleton_axes_at=[0, 0],
    # length=40,
)

b4mu_1_lf = TensorInfo(
    name="lf",
    root="GHUFNAGELLFLenseLeNet_Microscope",
    location="20191030_Beads_massiveGT/Beads_4micron/2019-10-30_08.34.35/stack_0_channel_0/TP_*/RC_rectified/Cam_Right_1_rectified.tif",
    tag="b4mu_1",
    insert_singleton_axes_at=[0, 0],
    # length=40,
)

b4mu_1_lr = TensorInfo(
    name="lr",
    root="GHUFNAGELLFLenseLeNet_Microscope",
    location="20191030_Beads_massiveGT/Beads_4micron/2019-10-30_08.34.35/stack_0_channel_0/TP_*/RCout/Cam_Right_1.tif",
    tag="b4mu_1",
    insert_singleton_axes_at=[0, 0],
    # length=40,
)

b4mu_1_ls = TensorInfo(
    name="ls_reg",
    root="GHUFNAGELLFLenseLeNet_Microscope",
    location="20191030_Beads_massiveGT/Beads_4micron/2019-10-30_08.34.35/stack_1_channel_1/TP_*/LC/Cam_Left_registered.tif",
    tag="b4mu_1",
    insert_singleton_axes_at=[0, 0],
    # length=40,
)

b4mu_2_lf = TensorInfo(
    name="lf",
    root="GHUFNAGELLFLenseLeNet_Microscope",
    location="20191030_Beads_massiveGT/Beads_4micron/2019-10-30_08.53.53/stack_0_channel_0/TP_*/RC_rectified/Cam_Right_1_rectified.tif",
    tag="b4mu_2",
    insert_singleton_axes_at=[0, 0],
    # length=40,
)

b4mu_2_lr = TensorInfo(
    name="lr",
    root="GHUFNAGELLFLenseLeNet_Microscope",
    location="20191030_Beads_massiveGT/Beads_4micron/2019-10-30_08.53.53/stack_0_channel_0/TP_*/RCout/Cam_Right_1.tif",
    tag="b4mu_2",
    insert_singleton_axes_at=[0, 0],
    # length=40,
)

b4mu_2_ls = TensorInfo(
    name="ls_reg",
    root="GHUFNAGELLFLenseLeNet_Microscope",
    location="20191030_Beads_massiveGT/Beads_4micron/2019-10-30_08.53.53/stack_1_channel_1/TP_*/LC/Cam_Left_registered.tif",
    tag="b4mu_2",
    insert_singleton_axes_at=[0, 0],
    # length=40,
)


b4mu_3_lf = TensorInfo(
    name="lf",
    root="GHUFNAGELLFLenseLeNet_Microscope",
    location="20191030_Beads_massiveGT/Beads_4micron/2019-10-30_09.14.05/stack_0_channel_0/TP_*/RC_rectified/Cam_Right_1_rectified.tif",
    tag="b4mu_3",
    insert_singleton_axes_at=[0, 0],
    # length=40,
)

b4mu_3_lr = TensorInfo(
    name="lr",
    root="GHUFNAGELLFLenseLeNet_Microscope",
    location="20191030_Beads_massiveGT/Beads_4micron/2019-10-30_09.14.05/stack_0_channel_0/TP_*/RCout/Cam_Right_1.tif",
    tag="b4mu_3",
    insert_singleton_axes_at=[0, 0],
    # length=40,
)

b4mu_3_ls = TensorInfo(
    name="ls_reg",
    root="GHUFNAGELLFLenseLeNet_Microscope",
    location="20191030_Beads_massiveGT/Beads_4micron/2019-10-30_09.14.05/stack_1_channel_1/TP_*/LC/Cam_Left_registered.tif",
    tag="b4mu_3",
    insert_singleton_axes_at=[0, 0],
    # length=40,
)

b01mix4_0_lf = TensorInfo(
    name="lf",
    root="GHUFNAGELLFLenseLeNet_Microscope",
    location="20191031_Beads_MixedSizes/Beads_mixed01and4microns/2019-10-31_02.27.33/stack_0_channel_0/TP_*/RC_rectified/Cam_Right_1_rectified.tif",
    tag="b01mix4_0",
    insert_singleton_axes_at=[0, 0],
)

b01mix4_0_lr = TensorInfo(
    name="lr",
    root="GHUFNAGELLFLenseLeNet_Microscope",
    location="20191031_Beads_MixedSizes/Beads_mixed01and4microns/2019-10-31_02.27.33/stack_0_channel_0/TP_*/RCout/Cam_Right_1.tif",
    tag="b01mix4_0",
    insert_singleton_axes_at=[0, 0],
)

b01mix4_0_ls = TensorInfo(
    name="ls_reg",
    root="GHUFNAGELLFLenseLeNet_Microscope",
    location="20191031_Beads_MixedSizes/Beads_mixed01and4microns/2019-10-31_02.27.33/stack_1_channel_1/TP_*/LC/Cam_Left_registered.tif",
    tag="b01mix4_0",
    insert_singleton_axes_at=[0, 0],
)

if __name__ == "__main__":
    from hashlib import sha224 as hash_algorithm

    from hylfm.datasets.base import get_dataset_from_info, N5CachedDatasetFromInfo

    # info = b4mu_3_ls
    for info in [b4mu_0_ls, b4mu_1_ls, b4mu_2_ls, b4mu_3_ls]:
        info.transformations += [
            {
                "Resize": {
                    "apply_to": "ls",
                    "shape": [
                        1.0,
                        0.14439140811455847255369928400955,
                        0.42105263157894736842105263157895,
                        0.42105263157894736842105263157895,
                    ],
                    "order": 2,
                }
            },
            {"Assert": {"apply_to": "ls", "expected_tensor_shape": [None, 1, 121, None, None]}},
        ]
        ds = get_dataset_from_info(info)
        print(
            settings.cache_dir
            / f"{ds.info.tag}_{ds.tensor_name}_{hash_algorithm(ds.description.encode()).hexdigest()}.n5"
        )
        ds = N5CachedDatasetFromInfo(ds)

        print(len(ds))
        print(ds[0]["ls"].shape)


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
