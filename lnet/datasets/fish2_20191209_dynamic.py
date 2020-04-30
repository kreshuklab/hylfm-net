from lnet.datasets.base import TensorInfo

t0454c2_lf = TensorInfo(
    name="lf",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_04.54.38/stack_1_channel_2/TP_*/RC_rectified/Cam_Right_*_rectified.tif",
    insert_singleton_axes_at=[0, 0],
    tag="t0454c2",
    z_slice="idx%241",
    # length=241 * 51,
)
t0454c2_ls = TensorInfo(
    name="ls",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_04.54.38/stack_1_channel_2/TP_*/LC/Cam_Left_*.tif",
    insert_singleton_axes_at=[0, 0, 0],
    tag="t0454c2",
    z_slice="idx%241",
    # length=241 * 51,
)

t0454c3_lf = TensorInfo(
    name="lf",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_04.54.38/stack_1_channel_3/TP_*/RC_rectified/Cam_Right_*_rectified.tif",
    tag="t0454c3",
    z_slice="idx%241",
    # length=241 * 51,
)
t0454c3_ls = TensorInfo(
    name="ls",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_04.54.38/stack_1_channel_3/TP_*/LC/Cam_Left_*.tif",
    tag="t0454c3",
    z_slice="idx%241",
    # length=241 * 51,
)


t0454c3_TP_00_lf = TensorInfo(
    name="lf",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_04.54.38/stack_1_channel_3/TP_00000/RC_rectified/Cam_Right_*_rectified.tif",
    tag="t0454c3_TP_00",
    z_slice="idx%241",
    # length=241,
)
t0454c3_TP_00_ls = TensorInfo(
    name="ls",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_04.54.38/stack_1_channel_3/TP_00000/LC/Cam_Left_*.tif",
    tag="t0454c3_TP_00",
    z_slice="idx%241",
    # length=241,
)


t0521c2_lf = TensorInfo(
    name="lf",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.21.16/stack_1_channel_2/TP_*/RC_rectified/Cam_Right_*_rectified.tif",
    tag="t0521c2",
    z_slice="idx%241",
    # length=241 * 24,
)
t0521c2_ls = TensorInfo(
    name="ls",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.21.16/stack_1_channel_2/TP_*/LC/Cam_Left_*.tif",
    tag="t0521c2",
    z_slice="idx%241",
    # length=241 * 24,
)

t0521c3_lf = TensorInfo(
    name="lf",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.21.16/stack_1_channel_3/TP_*/RC_rectified/Cam_Right_*_rectified.tif",
    tag="t0521c3",
    z_slice="idx%241",
    # length=241 * 25,
)
t0521c3_ls = TensorInfo(
    name="ls",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.21.16/stack_1_channel_3/TP_*/LC/Cam_Left_*.tif",
    tag="t0521c3",
    z_slice="idx%241",
    # length=241 * 25,
)

# # t0454c3_lf = TensorInfo(
# name = "lf",
# #     root = "KRESHUK",
# #     location="LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_04.54.38/stack_1_channel_3/TP_*/RC_rectified/Cam_Right_*_rectified.tif",
# #   tag=  "t0454",
# # #     length=241*?,
# #
# #     z_slice="idx%241",
# # t0454c3_lf = TensorInfo(
# name = "lf",
# #     root = "KRESHUK",
# #     location="LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_04.54.38/stack_1_channel_3/TP_*/RC_rectified/Cam_Right_*_rectified.tif",
# #   tag=  "t0454",
# # #     length=241*?,
# #
# #     z_slice="idx%241",
#
# # )
#
# # t0402c10_lf = TensorInfo(
# name = "lf",
# #     root = "KRESHUK",
# #     location="LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/fast_cropped_8ms/singlePlanes/plane_080/2019-12-09_04.02.24/stack_2_channel_10/TP_00000/RC_rectified/Cam_Right_*_rectified.tif",
# #   tag=  "t0402c10",
# # #     length=200,
# #     z_slice=241-80],
# # t0402c10_lf = TensorInfo(
# name = "lf",
# #     root = "KRESHUK",
# #     location="LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/fast_cropped_8ms/singlePlanes/plane_080/2019-12-09_04.02.24/stack_2_channel_10/TP_00000/RC_rectified/Cam_Right_*_rectified.tif",
# #   tag=  "t0402c10",
# # #     length=200,
# #     z_slice=241-80],
#
# #     z_slice="idx%241",
# # )

t0541c2p30_lf = TensorInfo(
    name="lf",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.41.14_theGoldenOne/singlePlane_samePos/plane_030/2019-12-09_06.07.03/stack_2_channel_2/TP_00000/RC_rectified/Cam_Right_*_rectified.tif",
    tag="t0541c2p30",
    z_slice=30,
    # length=200,
)

t0541c2p30_ls = TensorInfo(
    name="ls",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.41.14_theGoldenOne/singlePlane_samePos/plane_030/2019-12-09_06.07.03/stack_2_channel_2/TP_00000/LC/Cam_Left_*.tif",
    tag="t0541c2p30",
    z_slice=30,
    # length=200,
)
t0541c2p40_lf = TensorInfo(
    name="lf",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.41.14_theGoldenOne/singlePlane_samePos/plane_040/2019-12-09_06.05.50/stack_2_channel_2/TP_00000/RC_rectified/Cam_Right_*_rectified.tif",
    tag="t0541c2p40",
    z_slice=40,
    # length=200,
)

t0541c2p40_ls = TensorInfo(
    name="ls",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.41.14_theGoldenOne/singlePlane_samePos/plane_040/2019-12-09_06.05.50/stack_2_channel_2/TP_00000/LC/Cam_Left_*.tif",
    tag="t0541c2p40",
    z_slice=40,
    # length=200,
)
t0541c2p50_lf = TensorInfo(
    name="lf",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.41.14_theGoldenOne/singlePlane_samePos/plane_050/2019-12-09_06.15.21/stack_2_channel_2/TP_00000/RC_rectified/Cam_Right_*_rectified.tif",
    tag="t0541c2p50",
    z_slice=50,
    # length=200,
)

t0541c2p50_ls = TensorInfo(
    name="ls",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.41.14_theGoldenOne/singlePlane_samePos/plane_050/2019-12-09_06.15.21/stack_2_channel_2/TP_00000/LC/Cam_Left_*.tif",
    tag="t0541c2p50",
    z_slice=50,
    # length=200,
)
t0541c2p60_lf = TensorInfo(
    name="lf",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.41.14_theGoldenOne/singlePlane_samePos/plane_060/2019-12-09_06.01.42/stack_2_channel_2/TP_00000/RC_rectified/Cam_Right_*_rectified.tif",
    tag="t0541c2p60",
    z_slice=60,
    # length=200,
)

t0541c2p60_ls = TensorInfo(
    name="ls",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.41.14_theGoldenOne/singlePlane_samePos/plane_060/2019-12-09_06.01.42/stack_2_channel_2/TP_00000/LC/Cam_Left_*.tif",
    tag="t0541c2p60",
    z_slice=60,
    # length=200,
)
t0541c2p70_lf = TensorInfo(
    name="lf",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.41.14_theGoldenOne/singlePlane_samePos/plane_070/2019-12-09_06.14.09/stack_2_channel_2/TP_00000/RC_rectified/Cam_Right_*_rectified.tif",
    tag="t0541c2p70",
    z_slice=70,
    # length=200,
)

t0541c2p70_ls = TensorInfo(
    name="ls",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.41.14_theGoldenOne/singlePlane_samePos/plane_070/2019-12-09_06.14.09/stack_2_channel_2/TP_00000/LC/Cam_Left_*.tif",
    tag="t0541c2p70",
    z_slice=70,
    # length=200,
)
t0541c2p80_lf = TensorInfo(
    name="lf",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.41.14_theGoldenOne/singlePlane_samePos/plane_080/2019-12-09_05.58.38/stack_2_channel_2/TP_00000/RC_rectified/Cam_Right_*_rectified.tif",
    tag="t0541c2p80",
    z_slice=80,
    # length=200,
)

t0541c2p80_ls = TensorInfo(
    name="ls",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.41.14_theGoldenOne/singlePlane_samePos/plane_080/2019-12-09_05.58.38/stack_2_channel_2/TP_00000/LC/Cam_Left_*.tif",
    tag="t0541c2p80",
    z_slice=80,
    # length=200,
)

t0541c2p90_lf = TensorInfo(
    name="lf",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.41.14_theGoldenOne/singlePlane_samePos/plane_090/2019-12-09_06.13.02/stack_2_channel_2/TP_00000/RC_rectified/Cam_Right_*_rectified.tif",
    tag="t0541c2p90",
    z_slice=90,
    # length=200,
)

t0541c2p90_ls = TensorInfo(
    name="ls",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.41.14_theGoldenOne/singlePlane_samePos/plane_090/2019-12-09_06.13.02/stack_2_channel_2/TP_00000/LC/Cam_Left_*.tif",
    tag="t0541c2p90",
    z_slice=90,
    # length=200,
)
t0541c2p100_lf = TensorInfo(
    name="lf",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.41.14_theGoldenOne/singlePlane_samePos/plane_100/2019-12-09_05.55.26/stack_2_channel_2/TP_00000/RC_rectified/Cam_Right_*_rectified.tif",
    tag="t0541c2p100",
    z_slice=100,
    # length=200,
)

t0541c2p100_ls = TensorInfo(
    name="ls",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.41.14_theGoldenOne/singlePlane_samePos/plane_100/2019-12-09_05.55.26/stack_2_channel_2/TP_00000/LC/Cam_Left_*.tif",
    tag="t0541c2p100",
    z_slice=100,
    # length=200,
)
t0541c2p110_lf = TensorInfo(
    name="lf",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.41.14_theGoldenOne/singlePlane_samePos/plane_110/2019-12-09_06.08.14/stack_2_channel_2/TP_00000/RC_rectified/Cam_Right_*_rectified.tif",
    tag="t0541c2p110",
    z_slice=110,
    # length=200,
)

t0541c2p110_ls = TensorInfo(
    name="ls",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.41.14_theGoldenOne/singlePlane_samePos/plane_110/2019-12-09_06.08.14/stack_2_channel_2/TP_00000/LC/Cam_Left_*.tif",
    tag="t0541c2p110",
    z_slice=110,
    # length=200,
)
t0541c2p120_lf = TensorInfo(
    name="lf",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.41.14_theGoldenOne/singlePlane_samePos/plane_120/2019-12-09_05.53.55/stack_2_channel_2/TP_00000/RC_rectified/Cam_Right_*_rectified.tif",
    tag="t0541c2p120",
    z_slice=120,
    # length=200,
)

t0541c2p120_ls = TensorInfo(
    name="ls",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.41.14_theGoldenOne/singlePlane_samePos/plane_120/2019-12-09_05.53.55/stack_2_channel_2/TP_00000/LC/Cam_Left_*.tif",
    tag="t0541c2p120",
    z_slice=120,
    # length=200,
)
t0541c2p130_lf = TensorInfo(
    name="lf",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.41.14_theGoldenOne/singlePlane_samePos/plane_130/2019-12-09_06.09.31/stack_2_channel_2/TP_00000/RC_rectified/Cam_Right_*_rectified.tif",
    tag="t0541c2p130",
    z_slice=130,
    # length=200,
)

t0541c2p130_ls = TensorInfo(
    name="ls",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.41.14_theGoldenOne/singlePlane_samePos/plane_130/2019-12-09_06.09.31/stack_2_channel_2/TP_00000/LC/Cam_Left_*.tif",
    tag="t0541c2p130",
    z_slice=130,
    # length=200,
)
t0541c2p140_lf = TensorInfo(
    name="lf",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.41.14_theGoldenOne/singlePlane_samePos/plane_140/2019-12-09_06.00.01/stack_2_channel_2/TP_00000/RC_rectified/Cam_Right_*_rectified.tif",
    tag="t0541c2p140",
    z_slice=140,
    # length=200,
)

t0541c2p140_ls = TensorInfo(
    name="ls",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.41.14_theGoldenOne/singlePlane_samePos/plane_140/2019-12-09_06.00.01/stack_2_channel_2/TP_00000/LC/Cam_Left_*.tif",
    tag="t0541c2p140",
    z_slice=140,
    # length=200,
)
t0541c2p150_lf = TensorInfo(
    name="lf",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.41.14_theGoldenOne/singlePlane_samePos/plane_150/2019-12-09_06.10.38/stack_2_channel_2/TP_00000/RC_rectified/Cam_Right_*_rectified.tif",
    tag="t0541c2p150",
    z_slice=150,
    # length=200,
)

t0541c2p150_ls = TensorInfo(
    name="ls",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.41.14_theGoldenOne/singlePlane_samePos/plane_150/2019-12-09_06.10.38/stack_2_channel_2/TP_00000/LC/Cam_Left_*.tif",
    tag="t0541c2p150",
    z_slice=150,
    # length=200,
)
t0541c2p160_lf = TensorInfo(
    name="lf",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.41.14_theGoldenOne/singlePlane_samePos/plane_160/2019-12-09_06.03.01/stack_2_channel_2/TP_00000/RC_rectified/Cam_Right_*_rectified.tif",
    tag="t0541c2p160",
    z_slice=160,
    # length=200,
)

t0541c2p160_ls = TensorInfo(
    name="ls",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.41.14_theGoldenOne/singlePlane_samePos/plane_160/2019-12-09_06.03.01/stack_2_channel_2/TP_00000/LC/Cam_Left_*.tif",
    tag="t0541c2p160",
    z_slice=160,
    # length=200,
)
t0541c2p170_lf = TensorInfo(
    name="lf",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.41.14_theGoldenOne/singlePlane_samePos/plane_170/2019-12-09_06.11.51/stack_2_channel_2/TP_00000/RC_rectified/Cam_Right_*_rectified.tif",
    tag="t0541c2p170",
    z_slice=170,
    # length=200,
)

t0541c2p170_ls = TensorInfo(
    name="ls",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.41.14_theGoldenOne/singlePlane_samePos/plane_170/2019-12-09_06.11.51/stack_2_channel_2/TP_00000/LC/Cam_Left_*.tif",
    tag="t0541c2p170",
    z_slice=170,
    # length=200,
)
t0541c2p180_lf = TensorInfo(
    name="lf",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.41.14_theGoldenOne/singlePlane_samePos/plane_180/2019-12-09_06.04.27/stack_2_channel_2/TP_00000/RC_rectified/Cam_Right_*_rectified.tif",
    tag="t0541c2p180",
    z_slice=180,
    # length=200,
)

t0541c2p180_ls = TensorInfo(
    name="ls",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.41.14_theGoldenOne/singlePlane_samePos/plane_180/2019-12-09_06.04.27/stack_2_channel_2/TP_00000/LC/Cam_Left_*.tif",
    tag="t0541c2p180",
    z_slice=180,
    # length=200,
)

t0541c3p30_lf = TensorInfo(
    name="lf",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.41.14_theGoldenOne/singlePlane_samePos/plane_030/2019-12-09_06.07.03/stack_2_channel_3/TP_00000/RC_rectified/Cam_Right_*_rectified.tif",
    tag="t0541c3p30",
    z_slice=30,
    # length=200,
)

t0541c3p30_ls = TensorInfo(
    name="ls",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.41.14_theGoldenOne/singlePlane_samePos/plane_030/2019-12-09_06.07.03/stack_2_channel_3/TP_00000/LC/Cam_Left_*.tif",
    tag="t0541c3p30",
    z_slice=30,
    # length=200,
)
t0541c3p40_lf = TensorInfo(
    name="lf",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.41.14_theGoldenOne/singlePlane_samePos/plane_040/2019-12-09_06.05.50/stack_2_channel_3/TP_00000/RC_rectified/Cam_Right_*_rectified.tif",
    tag="t0541c3p40",
    z_slice=40,
    # length=200,
)

t0541c3p40_ls = TensorInfo(
    name="ls",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.41.14_theGoldenOne/singlePlane_samePos/plane_040/2019-12-09_06.05.50/stack_2_channel_3/TP_00000/LC/Cam_Left_*.tif",
    tag="t0541c3p40",
    z_slice=40,
    # length=200,
)
t0541c3p50_lf = TensorInfo(
    name="lf",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.41.14_theGoldenOne/singlePlane_samePos/plane_050/2019-12-09_06.15.21/stack_2_channel_3/TP_00000/RC_rectified/Cam_Right_*_rectified.tif",
    tag="t0541c3p50",
    z_slice=50,
    # length=200,
)

t0541c3p50_ls = TensorInfo(
    name="ls",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.41.14_theGoldenOne/singlePlane_samePos/plane_050/2019-12-09_06.15.21/stack_2_channel_3/TP_00000/LC/Cam_Left_*.tif",
    tag="t0541c3p50",
    z_slice=50,
    # length=200,
)
t0541c3p60_lf = TensorInfo(
    name="lf",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.41.14_theGoldenOne/singlePlane_samePos/plane_060/2019-12-09_06.01.42/stack_2_channel_3/TP_00000/RC_rectified/Cam_Right_*_rectified.tif",
    tag="t0541c3p60",
    z_slice=60,
    # length=200,
)

t0541c3p60_lf = TensorInfo(
    name="lf",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.41.14_theGoldenOne/singlePlane_samePos/plane_060/2019-12-09_06.01.42/stack_2_channel_3/TP_00000/RC_rectified/Cam_Right_*_rectified.tif",
    tag="t0541c3p60",
    z_slice=60,
    # length=200,
)
t0541c3p70_lf = TensorInfo(
    name="lf",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.41.14_theGoldenOne/singlePlane_samePos/plane_070/2019-12-09_06.14.09/stack_2_channel_3/TP_00000/RC_rectified/Cam_Right_*_rectified.tif",
    tag="t0541c3p70",
    z_slice=70,
    # length=200,
)

t0541c3p70_ls = TensorInfo(
    name="ls",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.41.14_theGoldenOne/singlePlane_samePos/plane_070/2019-12-09_06.14.09/stack_2_channel_3/TP_00000/LC/Cam_Left_*.tif",
    tag="t0541c3p70",
    z_slice=70,
    # length=200,
)
t0541c3p80_lf = TensorInfo(
    name="lf",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.41.14_theGoldenOne/singlePlane_samePos/plane_080/2019-12-09_05.58.38/stack_2_channel_3/TP_00000/RC_rectified/Cam_Right_*_rectified.tif",
    tag="t0541c3p80",
    z_slice=80,
    # length=200,
)

t0541c3p80_ls = TensorInfo(
    name="ls",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.41.14_theGoldenOne/singlePlane_samePos/plane_080/2019-12-09_05.58.38/stack_2_channel_3/TP_00000/LC/Cam_Left_*.tif",
    tag="t0541c3p80",
    z_slice=80,
    # length=200,
)

t0541c3p90_lf = TensorInfo(
    name="lf",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.41.14_theGoldenOne/singlePlane_samePos/plane_090/2019-12-09_06.13.02/stack_2_channel_3/TP_00000/RC_rectified/Cam_Right_*_rectified.tif",
    tag="t0541c3p90",
    z_slice=90,
    # length=200,
)

t0541c3p90_ls = TensorInfo(
    name="ls",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.41.14_theGoldenOne/singlePlane_samePos/plane_090/2019-12-09_06.13.02/stack_2_channel_3/TP_00000/LC/Cam_Left_*.tif",
    tag="t0541c3p90",
    z_slice=90,
    # length=200,
)
t0541c3p100_lf = TensorInfo(
    name="lf",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.41.14_theGoldenOne/singlePlane_samePos/plane_100/2019-12-09_05.55.26/stack_2_channel_3/TP_00000/RC_rectified/Cam_Right_*_rectified.tif",
    tag="t0541c3p100",
    z_slice=100,
    # length=200,
)

t0541c3p100_ls = TensorInfo(
    name="ls",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.41.14_theGoldenOne/singlePlane_samePos/plane_100/2019-12-09_05.55.26/stack_2_channel_3/TP_00000/LC/Cam_Left_*.tif",
    tag="t0541c3p100",
    z_slice=100,
    # length=200,
)
t0541c3p110_lf = TensorInfo(
    name="lf",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.41.14_theGoldenOne/singlePlane_samePos/plane_110/2019-12-09_06.08.14/stack_2_channel_3/TP_00000/RC_rectified/Cam_Right_*_rectified.tif",
    tag="t0541c3p110",
    z_slice=110,
    # length=200,
)

t0541c3p110_ls = TensorInfo(
    name="ls",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.41.14_theGoldenOne/singlePlane_samePos/plane_110/2019-12-09_06.08.14/stack_2_channel_3/TP_00000/LC/Cam_Left_*.tif",
    tag="t0541c3p110",
    z_slice=110,
    # length=200,
)
t0541c3p120_lf = TensorInfo(
    name="lf",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.41.14_theGoldenOne/singlePlane_samePos/plane_120/2019-12-09_05.53.55/stack_2_channel_3/TP_00000/RC_rectified/Cam_Right_*_rectified.tif",
    tag="t0541c3p120",
    z_slice=120,
    # length=200,
)

t0541c3p120_ls = TensorInfo(
    name="ls",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.41.14_theGoldenOne/singlePlane_samePos/plane_120/2019-12-09_05.53.55/stack_2_channel_3/TP_00000/LC/Cam_Left_*.tif",
    tag="t0541c3p120",
    z_slice=120,
    # length=200,
)
t0541c3p130_lf = TensorInfo(
    name="lf",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.41.14_theGoldenOne/singlePlane_samePos/plane_130/2019-12-09_06.09.31/stack_2_channel_3/TP_00000/RC_rectified/Cam_Right_*_rectified.tif",
    tag="t0541c3p130",
    z_slice=130,
    # length=200,
)

t0541c3p130_ls = TensorInfo(
    name="ls",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.41.14_theGoldenOne/singlePlane_samePos/plane_130/2019-12-09_06.09.31/stack_2_channel_3/TP_00000/LC/Cam_Left_*.tif",
    tag="t0541c3p130",
    z_slice=130,
    # length=200,
)
t0541c3p140_lf = TensorInfo(
    name="lf",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.41.14_theGoldenOne/singlePlane_samePos/plane_140/2019-12-09_06.00.01/stack_2_channel_3/TP_00000/RC_rectified/Cam_Right_*_rectified.tif",
    tag="t0541c3p140",
    z_slice=140,
    # length=200,
)

t0541c3p140_ls = TensorInfo(
    name="ls",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.41.14_theGoldenOne/singlePlane_samePos/plane_140/2019-12-09_06.00.01/stack_2_channel_3/TP_00000/LC/Cam_Left_*.tif",
    tag="t0541c3p140",
    z_slice=140,
    # length=200,
)
t0541c3p150_lf = TensorInfo(
    name="lf",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.41.14_theGoldenOne/singlePlane_samePos/plane_150/2019-12-09_06.10.38/stack_2_channel_3/TP_00000/RC_rectified/Cam_Right_*_rectified.tif",
    tag="t0541c3p150",
    z_slice=150,
    # length=200,
)

t0541c3p150_ls = TensorInfo(
    name="ls",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.41.14_theGoldenOne/singlePlane_samePos/plane_150/2019-12-09_06.10.38/stack_2_channel_3/TP_00000/LC/Cam_Left_*.tif",
    tag="t0541c3p150",
    z_slice=150,
    # length=200,
)
t0541c3p160_lf = TensorInfo(
    name="lf",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.41.14_theGoldenOne/singlePlane_samePos/plane_160/2019-12-09_06.03.01/stack_2_channel_3/TP_00000/RC_rectified/Cam_Right_*_rectified.tif",
    tag="t0541c3p160",
    z_slice=160,
    # length=200,
)

t0541c3p160_ls = TensorInfo(
    name="ls",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.41.14_theGoldenOne/singlePlane_samePos/plane_160/2019-12-09_06.03.01/stack_2_channel_3/TP_00000/LC/Cam_Left_*.tif",
    tag="t0541c3p160",
    z_slice=160,
    # length=200,
)
t0541c3p170_lf = TensorInfo(
    name="lf",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.41.14_theGoldenOne/singlePlane_samePos/plane_170/2019-12-09_06.11.51/stack_2_channel_3/TP_00000/RC_rectified/Cam_Right_*_rectified.tif",
    tag="t0541c3p170",
    z_slice=170,
    # length=200,
)

t0541c3p170_ls = TensorInfo(
    name="ls",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.41.14_theGoldenOne/singlePlane_samePos/plane_170/2019-12-09_06.11.51/stack_2_channel_3/TP_00000/LC/Cam_Left_*.tif",
    tag="t0541c3p170",
    z_slice=170,
    # length=200,
)
t0541c3p180_lf = TensorInfo(
    name="lf",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.41.14_theGoldenOne/singlePlane_samePos/plane_180/2019-12-09_06.04.27/stack_2_channel_3/TP_00000/RC_rectified/Cam_Right_*_rectified.tif",
    tag="t0541c3p180",
    z_slice=180,
    # length=200,
)

t0541c3p180_ls = TensorInfo(
    name="ls",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.41.14_theGoldenOne/singlePlane_samePos/plane_180/2019-12-09_06.04.27/stack_2_channel_3/TP_00000/LC/Cam_Left_*.tif",
    tag="t0541c3p180",
    z_slice=180,
    # length=200,
)


t0402c10p80a_lf = TensorInfo(
    name="lf",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/fast_cropped_8ms/singlePlanes/plane_080/2019-12-09_04.02.24/stack_2_channel_10/TP_00000/RC_rectified/Cam_Right_*_rectified.tif",
    tag="t0402c10p80a",
    z_slice=80,
    # length=200,
)

t0402c10p80a_ls = TensorInfo(
    name="ls",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/fast_cropped_8ms/singlePlanes/plane_080/2019-12-09_04.02.24/stack_2_channel_10/TP_00000/LC/Cam_Left_*.tif",
    tag="t0402c10p80a",
    z_slice=80,
    # length=200,
)
t0402c11p80a_lf = TensorInfo(
    name="lf",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/fast_cropped_8ms/singlePlanes/plane_080/2019-12-09_04.02.24/stack_2_channel_11/TP_00000/RC_rectified/Cam_Right_*_rectified.tif",
    tag="t0402c11p80a",
    z_slice=80,
    # length=200,
)

t0402c11p80a_ls = TensorInfo(
    name="ls",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/fast_cropped_8ms/singlePlanes/plane_080/2019-12-09_04.02.24/stack_2_channel_11/TP_00000/LC/Cam_Left_*.tif",
    tag="t0402c11p80a",
    z_slice=80,
    # length=200,
)
t0402c10p80b_lf = TensorInfo(
    name="lf",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/fast_cropped_8ms/singlePlanes/plane_080/2019-12-09_04.05.07_irisOpenedComplete/stack_2_channel_10/TP_00000/RC_rectified/Cam_Right_*_rectified.tif",
    tag="t0402c10p80b",
    z_slice=80,
    # length=200,
)

t0402c10p80b_ls = TensorInfo(
    name="ls",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/fast_cropped_8ms/singlePlanes/plane_080/2019-12-09_04.05.07_irisOpenedComplete/stack_2_channel_10/TP_00000/LC/Cam_Left_*.tif",
    tag="t0402c10p80b",
    z_slice=80,
    # length=200,
)
t0402c11p80b_lf = TensorInfo(
    name="lf",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/fast_cropped_8ms/singlePlanes/plane_080/2019-12-09_04.05.07_irisOpenedComplete/stack_2_channel_11/TP_00000/RC_rectified/Cam_Right_*_rectified.tif",
    tag="t0402c11p80b",
    z_slice=80,
    # length=200,
)

t0402c11p80b_ls = TensorInfo(
    name="ls",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/fast_cropped_8ms/singlePlanes/plane_080/2019-12-09_04.05.07_irisOpenedComplete/stack_2_channel_11/TP_00000/LC/Cam_Left_*.tif",
    tag="t0402c11p80b",
    z_slice=80,
    # length=200,
)
t0402c10p100a_lf = TensorInfo(
    name="lf",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/fast_cropped_8ms/singlePlanes/plane_100/2019-12-09_03.44.34/stack_2_channel_10/TP_00000/RC_rectified/Cam_Right_*_rectified.tif",
    tag="t0402c10p100a",
    z_slice=100,
    # length=200,
)

t0402c10p100a_ls = TensorInfo(
    name="ls",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/fast_cropped_8ms/singlePlanes/plane_100/2019-12-09_03.44.34/stack_2_channel_10/TP_00000/LC/Cam_Left_*.tif",
    tag="t0402c10p100a",
    z_slice=100,
    # length=200,
)
t0402c11p100a_lf = TensorInfo(
    name="lf",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/fast_cropped_8ms/singlePlanes/plane_100/2019-12-09_03.44.34/stack_2_channel_11/TP_00000/RC_rectified/Cam_Right_*_rectified.tif",
    tag="t0402c11p100a",
    z_slice=100,
    # length=200,
)

t0402c11p100a_ls = TensorInfo(
    name="ls",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/fast_cropped_8ms/singlePlanes/plane_100/2019-12-09_03.44.34/stack_2_channel_11/TP_00000/LC/Cam_Left_*.tif",
    tag="t0402c11p100a",
    z_slice=100,
    # length=200,
)
t0402c10p100b_lf = TensorInfo(
    name="lf",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/fast_cropped_8ms/singlePlanes/plane_100/2019-12-09_03.46.56/stack_2_channel_10/TP_00000/RC_rectified/Cam_Right_*_rectified.tif",
    tag="t0402c10p100b",
    z_slice=100,
    # length=200,
)

t0402c10p100b_ls = TensorInfo(
    name="ls",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/fast_cropped_8ms/singlePlanes/plane_100/2019-12-09_03.46.56/stack_2_channel_10/TP_00000/LC/Cam_Left_*.tif",
    tag="t0402c10p100b",
    z_slice=100,
    # length=200,
)
t0402c11p100b_lf = TensorInfo(
    name="lf",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/fast_cropped_8ms/singlePlanes/plane_100/2019-12-09_03.46.56/stack_2_channel_11/TP_00000/RC_rectified/Cam_Right_*_rectified.tif",
    tag="t0402c11p100b",
    z_slice=100,
    # length=200,
)

t0402c11p100b_ls = TensorInfo(
    name="ls",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/fast_cropped_8ms/singlePlanes/plane_100/2019-12-09_03.46.56/stack_2_channel_11/TP_00000/LC/Cam_Left_*.tif",
    tag="t0402c11p100b",
    z_slice=100,
    # length=200,
)
t0402c10p120a_lf = TensorInfo(
    name="lf",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/fast_cropped_8ms/singlePlanes/plane_120/2019-12-09_03.41.23/stack_2_channel_10/TP_00000/RC_rectified/Cam_Right_*_rectified.tif",
    tag="t0402c10p120a",
    z_slice=120,
    # length=200,
)

t0402c10p120a_ls = TensorInfo(
    name="ls",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/fast_cropped_8ms/singlePlanes/plane_120/2019-12-09_03.41.23/stack_2_channel_10/TP_00000/LC/Cam_Left_*.tif",
    tag="t0402c10p120a",
    z_slice=120,
    # length=200,
)
t0402c11p120a_lf = TensorInfo(
    name="lf",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/fast_cropped_8ms/singlePlanes/plane_120/2019-12-09_03.41.23/stack_2_channel_11/TP_00000/RC_rectified/Cam_Right_*_rectified.tif",
    tag="t0402c11p120a",
    z_slice=120,
    # length=200,
)

t0402c11p120a_ls = TensorInfo(
    name="ls",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/fast_cropped_8ms/singlePlanes/plane_120/2019-12-09_03.41.23/stack_2_channel_11/TP_00000/LC/Cam_Left_*.tif",
    tag="t0402c11p120a",
    z_slice=120,
    # length=200,
)
t0402c10p120b_lf = TensorInfo(
    name="lf",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/fast_cropped_8ms/singlePlanes/plane_120/2019-12-09_03.42.18/stack_2_channel_10/TP_00000/RC_rectified/Cam_Right_*_rectified.tif",
    tag="t0402c10p120b",
    z_slice=120,
    # length=200,
)

t0402c10p120b_ls = TensorInfo(
    name="ls",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/fast_cropped_8ms/singlePlanes/plane_120/2019-12-09_03.42.18/stack_2_channel_10/TP_00000/LC/Cam_Left_*.tif",
    tag="t0402c10p120b",
    z_slice=120,
    # length=200,
)
t0402c11p120b_lf = TensorInfo(
    name="lf",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/fast_cropped_8ms/singlePlanes/plane_120/2019-12-09_03.42.18/stack_2_channel_11/TP_00000/RC_rectified/Cam_Right_*_rectified.tif",
    tag="t0402c11p120b",
    z_slice=120,
    # length=200,
)
t0402c11p120b_ls = TensorInfo(
    name="ls",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/fast_cropped_8ms/singlePlanes/plane_120/2019-12-09_03.42.18/stack_2_channel_11/TP_00000/LC/Cam_Left_*.tif",
    tag="t0402c11p120b",
    z_slice=120,
    # length=200,
)
t0402c10p140a_lf = TensorInfo(
    name="lf",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/fast_cropped_8ms/singlePlanes/plane_140/2019-12-09_03.55.51/stack_2_channel_10/TP_00000/RC_rectified/Cam_Right_*_rectified.tif",
    tag="t0402c10p140a",
    z_slice=140,
    # length=200,
)

t0402c10p140a_ls = TensorInfo(
    name="ls",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/fast_cropped_8ms/singlePlanes/plane_140/2019-12-09_03.55.51/stack_2_channel_10/TP_00000/LC/Cam_Left_*.tif",
    tag="t0402c10p140a",
    z_slice=140,
    # length=200,
)
t0402c11p140a_lf = TensorInfo(
    name="lf",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/fast_cropped_8ms/singlePlanes/plane_140/2019-12-09_03.55.51/stack_2_channel_11/TP_00000/RC_rectified/Cam_Right_*_rectified.tif",
    tag="t0402c11p140a",
    z_slice=140,
    # length=200,
)

t0402c11p140a_ls = TensorInfo(
    name="ls",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/fast_cropped_8ms/singlePlanes/plane_140/2019-12-09_03.55.51/stack_2_channel_11/TP_00000/LC/Cam_Left_*.tif",
    tag="t0402c11p140a",
    z_slice=140,
    # length=200,
)
t0402c10p140b_lf = TensorInfo(
    name="lf",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/fast_cropped_8ms/singlePlanes/plane_140/2019-12-09_03.56.44/stack_2_channel_10/TP_00000/RC_rectified/Cam_Right_*_rectified.tif",
    tag="t0402c10p140b",
    z_slice=140,
    # length=200,
)

t0402c10p140b_ls = TensorInfo(
    name="ls",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/fast_cropped_8ms/singlePlanes/plane_140/2019-12-09_03.56.44/stack_2_channel_10/TP_00000/LC/Cam_Left_*.tif",
    tag="t0402c10p140b",
    z_slice=140,
    # length=200,
)
t0402c11p140b_lf = TensorInfo(
    name="lf",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/fast_cropped_8ms/singlePlanes/plane_140/2019-12-09_03.56.44/stack_2_channel_11/TP_00000/RC_rectified/Cam_Right_*_rectified.tif",
    tag="t0402c11p140b",
    z_slice=140,
    # length=200,
)

t0402c11p140b_ls = TensorInfo(
    name="ls",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/fast_cropped_8ms/singlePlanes/plane_140/2019-12-09_03.56.44/stack_2_channel_11/TP_00000/LC/Cam_Left_*.tif",
    tag="t0402c11p140b",
    z_slice=140,
    # length=200,
)
t0402c10p160a_lf = TensorInfo(
    name="lf",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/fast_cropped_8ms/singlePlanes/plane_160/2019-12-09_03.58.24/stack_2_channel_10/TP_00000/RC_rectified/Cam_Right_*_rectified.tif",
    tag="t0402c10p160a",
    z_slice=160,
    # length=200,
)

t0402c10p160a_ls = TensorInfo(
    name="ls",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/fast_cropped_8ms/singlePlanes/plane_160/2019-12-09_03.58.24/stack_2_channel_10/TP_00000/LC/Cam_Left_*.tif",
    tag="t0402c10p160a",
    z_slice=160,
    # length=200,
)
t0402c11p160a_lf = TensorInfo(
    name="lf",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/fast_cropped_8ms/singlePlanes/plane_160/2019-12-09_03.58.24/stack_2_channel_11/TP_00000/RC_rectified/Cam_Right_*_rectified.tif",
    tag="t0402c11p160a",
    z_slice=160,
    # length=200,
)

t0402c11p160a_ls = TensorInfo(
    name="ls",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/fast_cropped_8ms/singlePlanes/plane_160/2019-12-09_03.58.24/stack_2_channel_11/TP_00000/LC/Cam_Left_*.tif",
    tag="t0402c11p160a",
    z_slice=160,
    # length=200,
)
t0402c10p160b_lf = TensorInfo(
    name="lf",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/fast_cropped_8ms/singlePlanes/plane_160/2019-12-09_03.59.45/stack_2_channel_10/TP_00000/RC_rectified/Cam_Right_*_rectified.tif",
    tag="t0402c10p160b",
    z_slice=160,
    # length=200,
)

t0402c10p160b_ls = TensorInfo(
    name="ls",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/fast_cropped_8ms/singlePlanes/plane_160/2019-12-09_03.59.45/stack_2_channel_10/TP_00000/LC/Cam_Left_*.tif",
    tag="t0402c10p160b",
    z_slice=160,
    # length=200,
)
t0402c11p160b_lf = TensorInfo(
    name="lf",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/fast_cropped_8ms/singlePlanes/plane_160/2019-12-09_03.59.45/stack_2_channel_11/TP_00000/RC_rectified/Cam_Right_*_rectified.tif",
    tag="t0402c11p160b",
    z_slice=160,
    # length=200,
)

t0402c11p160b_ls = TensorInfo(
    name="ls",
    root="GKRESHUK",
    location="LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/fast_cropped_8ms/singlePlanes/plane_160/2019-12-09_03.59.45/stack_2_channel_11/TP_00000/LC/Cam_Left_*.tif",
    tag="t0402c11p160b",
    z_slice=160,
    # length=200,
)

if __name__ == "__main__":
    import logging

    from hashlib import sha224 as hash_algorithm
    from lnet.settings import settings
    from lnet.datasets.base import get_dataset_from_info, N5CachedDatasetFromInfo

    logger = logging.getLogger("gcamp")

    for info_lf, info_ls in zip(
        [
t0454c2_lf
        ],
        [
t0454c2_ls
        ],
    ):
        try:
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
        except Exception as e:
            print("error")
            logger.error(e, exc_info=True)
