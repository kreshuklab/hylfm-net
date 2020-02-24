from pathlib import Path

from lnet.config.dataset import NamedDatasetInfo

t0454c2 = NamedDatasetInfo(
    Path("/"),
    "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_04.54.38/stack_1_channel_2/TP_*/RC_rectified/Cam_Right_*_rectified.tif",
    "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_04.54.38/stack_1_channel_2/TP_*/LC/Cam_Left_*.tif",
    "t0454c2",
    length=241 * 51,
    AffineTransform="from_x_path",
    dynamic_z_slice_mod=241,
)
t0454c2_test = NamedDatasetInfo(
    Path("/"),
    "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_04.54.38/stack_1_channel_2/TP_*/RC_rectified/Cam_Right_*_rectified.tif",
    None,
    description="t0454c2_test",
    length=241 * 51,
    AffineTransform="from_x_path",
)
t0454c3 = NamedDatasetInfo(
    Path("/"),
    "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_04.54.38/stack_1_channel_3/TP_*/RC_rectified/Cam_Right_*_rectified.tif",
    "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_04.54.38/stack_1_channel_3/TP_*/LC/Cam_Left_*.tif",
    "t0454c3",
    length=241 * 51,
    AffineTransform="from_x_path",
    dynamic_z_slice_mod=241,
)
t0454c3_test = NamedDatasetInfo(
    Path("/"),
    "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_04.54.38/stack_1_channel_3/TP_*/RC_rectified/Cam_Right_*_rectified.tif",
    None,
    description="t0454c3_test",
    length=241 * 51,
    AffineTransform="from_x_path",
)


t0454c3_TP_00 = NamedDatasetInfo(
    Path("/"),
    "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_04.54.38/stack_1_channel_3/TP_00000/RC_rectified/Cam_Right_*_rectified.tif",
    "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_04.54.38/stack_1_channel_3/TP_00000/LC/Cam_Left_*.tif",
    "t0454c3_TP_00",
    length=241,
    AffineTransform="from_x_path",
    dynamic_z_slice_mod=241,
)

t0521c2 = NamedDatasetInfo(
    Path("/"),
    "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.21.16/stack_1_channel_2/TP_*/RC_rectified/Cam_Right_*_rectified.tif",
    "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.21.16/stack_1_channel_2/TP_*/LC/Cam_Left_*.tif",
    "t0521c2",
    length=241 * 24,
    AffineTransform="from_x_path",
    dynamic_z_slice_mod=241,
)
t0521c3 = NamedDatasetInfo(
    Path("/"),
    "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.21.16/stack_1_channel_3/TP_*/RC_rectified/Cam_Right_*_rectified.tif",
    "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.21.16/stack_1_channel_3/TP_*/LC/Cam_Left_*.tif",
    "t0521c3",
    length=241 * 25,
    AffineTransform="from_x_path",
    dynamic_z_slice_mod=241,
)
# t0454c3 = NamedDatasetInfo(
#     Path("/"),
#     "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_04.54.38/stack_1_channel_3/TP_*/RC_rectified/Cam_Right_*_rectified.tif",
#     "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_04.54.38/stack_1_channel_3/TP_*/LC/Cam_Left_*.tif",
#     "t0454",
#     length=241*?,
#     AffineTransform="from_x_path",
#     dynamic_z_slice_mod=241,
# )

# t0402c10 = NamedDatasetInfo(
#     Path("/"),
#     "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/fast_cropped_8ms/singlePlanes/plane_080/2019-12-09_04.02.24/stack_2_channel_10/TP_00000/RC_rectified/Cam_Right_*_rectified.tif",
#     "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/fast_cropped_8ms/singlePlanes/plane_080/2019-12-09_04.02.24/stack_2_channel_10/TP_00000/LC/Cam_Left_*.tif",
#     "t0402c10",
#     length=200,
#     AffineTransform="from_x_path",
#     z_slices=[24+1-80],
#     dynamic_z_slice_mod=241,
# )

t0541c2p30 = NamedDatasetInfo(
    Path("/"),
    "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.41.14_theGoldenOne/singlePlane_samePos/plane_030/2019-12-09_06.07.03/stack_2_channel_2/TP_00000/RC_rectified/Cam_Right_*_rectified.tif",
    "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.41.14_theGoldenOne/singlePlane_samePos/plane_030/2019-12-09_06.07.03/stack_2_channel_2/TP_00000/LC/Cam_Left_*.tif",
    "t0541c2p30",
    length=200,
    AffineTransform="from_x_path",
    dynamic_z_slice_mod=241,
    z_slices=[30],
)
t0541c2p40 = NamedDatasetInfo(
    Path("/"),
    "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.41.14_theGoldenOne/singlePlane_samePos/plane_040/2019-12-09_06.05.50/stack_2_channel_2/TP_00000/RC_rectified/Cam_Right_*_rectified.tif",
    "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.41.14_theGoldenOne/singlePlane_samePos/plane_040/2019-12-09_06.05.50/stack_2_channel_2/TP_00000/LC/Cam_Left_*.tif",
    "t0541c2p40",
    length=200,
    AffineTransform="from_x_path",
    dynamic_z_slice_mod=241,
    z_slices=[40],
)
t0541c2p50 = NamedDatasetInfo(
    Path("/"),
    "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.41.14_theGoldenOne/singlePlane_samePos/plane_050/2019-12-09_06.15.21/stack_2_channel_2/TP_00000/RC_rectified/Cam_Right_*_rectified.tif",
    "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.41.14_theGoldenOne/singlePlane_samePos/plane_050/2019-12-09_06.15.21/stack_2_channel_2/TP_00000/LC/Cam_Left_*.tif",
    "t0541c2p50",
    length=200,
    AffineTransform="from_x_path",
    dynamic_z_slice_mod=241,
    z_slices=[50],
)
t0541c2p60 = NamedDatasetInfo(
    Path("/"),
    "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.41.14_theGoldenOne/singlePlane_samePos/plane_060/2019-12-09_06.01.42/stack_2_channel_2/TP_00000/RC_rectified/Cam_Right_*_rectified.tif",
    "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.41.14_theGoldenOne/singlePlane_samePos/plane_060/2019-12-09_06.01.42/stack_2_channel_2/TP_00000/LC/Cam_Left_*.tif",
    "t0541c2p60",
    length=200,
    AffineTransform="from_x_path",
    dynamic_z_slice_mod=241,
    z_slices=[60],
)
t0541c2p70 = NamedDatasetInfo(
    Path("/"),
    "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.41.14_theGoldenOne/singlePlane_samePos/plane_070/2019-12-09_06.14.09/stack_2_channel_2/TP_00000/RC_rectified/Cam_Right_*_rectified.tif",
    "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.41.14_theGoldenOne/singlePlane_samePos/plane_070/2019-12-09_06.14.09/stack_2_channel_2/TP_00000/LC/Cam_Left_*.tif",
    "t0541c2p70",
    length=200,
    AffineTransform="from_x_path",
    dynamic_z_slice_mod=241,
    z_slices=[70],
)
t0541c2p80 = NamedDatasetInfo(
    Path("/"),
    "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.41.14_theGoldenOne/singlePlane_samePos/plane_080/2019-12-09_05.58.38/stack_2_channel_2/TP_00000/RC_rectified/Cam_Right_*_rectified.tif",
    "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.41.14_theGoldenOne/singlePlane_samePos/plane_080/2019-12-09_05.58.38/stack_2_channel_2/TP_00000/LC/Cam_Left_*.tif",
    "t0541c2p80",
    length=200,
    AffineTransform="from_x_path",
    dynamic_z_slice_mod=241,
    z_slices=[80],
)

t0541c2p90 = NamedDatasetInfo(
    Path("/"),
    "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.41.14_theGoldenOne/singlePlane_samePos/plane_090/2019-12-09_06.13.02/stack_2_channel_2/TP_00000/RC_rectified/Cam_Right_*_rectified.tif",
    "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.41.14_theGoldenOne/singlePlane_samePos/plane_090/2019-12-09_06.13.02/stack_2_channel_2/TP_00000/LC/Cam_Left_*.tif",
    "t0541c2p90",
    length=200,
    AffineTransform="from_x_path",
    dynamic_z_slice_mod=241,
    z_slices=[90],
)
t0541c2p100 = NamedDatasetInfo(
    Path("/"),
    "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.41.14_theGoldenOne/singlePlane_samePos/plane_100/2019-12-09_05.55.26/stack_2_channel_2/TP_00000/RC_rectified/Cam_Right_*_rectified.tif",
    "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.41.14_theGoldenOne/singlePlane_samePos/plane_100/2019-12-09_05.55.26/stack_2_channel_2/TP_00000/LC/Cam_Left_*.tif",
    "t0541c2p100",
    length=200,
    AffineTransform="from_x_path",
    dynamic_z_slice_mod=241,
    z_slices=[100],
)
t0541c2p110 = NamedDatasetInfo(
    Path("/"),
    "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.41.14_theGoldenOne/singlePlane_samePos/plane_110/2019-12-09_06.08.14/stack_2_channel_2/TP_00000/RC_rectified/Cam_Right_*_rectified.tif",
    "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.41.14_theGoldenOne/singlePlane_samePos/plane_110/2019-12-09_06.08.14/stack_2_channel_2/TP_00000/LC/Cam_Left_*.tif",
    "t0541c2p110",
    length=200,
    AffineTransform="from_x_path",
    dynamic_z_slice_mod=241,
    z_slices=[110],
)
t0541c2p120 = NamedDatasetInfo(
    Path("/"),
    "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.41.14_theGoldenOne/singlePlane_samePos/plane_120/2019-12-09_05.53.55/stack_2_channel_2/TP_00000/RC_rectified/Cam_Right_*_rectified.tif",
    "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.41.14_theGoldenOne/singlePlane_samePos/plane_120/2019-12-09_05.53.55/stack_2_channel_2/TP_00000/LC/Cam_Left_*.tif",
    "t0541c2p120",
    length=200,
    AffineTransform="from_x_path",
    dynamic_z_slice_mod=241,
    z_slices=[120],
)
t0541c2p130 = NamedDatasetInfo(
    Path("/"),
    "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.41.14_theGoldenOne/singlePlane_samePos/plane_130/2019-12-09_06.09.31/stack_2_channel_2/TP_00000/RC_rectified/Cam_Right_*_rectified.tif",
    "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.41.14_theGoldenOne/singlePlane_samePos/plane_130/2019-12-09_06.09.31/stack_2_channel_2/TP_00000/LC/Cam_Left_*.tif",
    "t0541c2p130",
    length=200,
    AffineTransform="from_x_path",
    dynamic_z_slice_mod=241,
    z_slices=[130],
)
t0541c2p140 = NamedDatasetInfo(
    Path("/"),
    "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.41.14_theGoldenOne/singlePlane_samePos/plane_140/2019-12-09_06.00.01/stack_2_channel_2/TP_00000/RC_rectified/Cam_Right_*_rectified.tif",
    "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.41.14_theGoldenOne/singlePlane_samePos/plane_140/2019-12-09_06.00.01/stack_2_channel_2/TP_00000/LC/Cam_Left_*.tif",
    "t0541c2p140",
    length=200,
    AffineTransform="from_x_path",
    dynamic_z_slice_mod=241,
    z_slices=[140],
)
t0541c2p150 = NamedDatasetInfo(
    Path("/"),
    "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.41.14_theGoldenOne/singlePlane_samePos/plane_150/2019-12-09_06.10.38/stack_2_channel_2/TP_00000/RC_rectified/Cam_Right_*_rectified.tif",
    "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.41.14_theGoldenOne/singlePlane_samePos/plane_150/2019-12-09_06.10.38/stack_2_channel_2/TP_00000/LC/Cam_Left_*.tif",
    "t0541c2p150",
    length=200,
    AffineTransform="from_x_path",
    dynamic_z_slice_mod=241,
    z_slices=[150],
)
t0541c2p160 = NamedDatasetInfo(
    Path("/"),
    "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.41.14_theGoldenOne/singlePlane_samePos/plane_160/2019-12-09_06.03.01/stack_2_channel_2/TP_00000/RC_rectified/Cam_Right_*_rectified.tif",
    "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.41.14_theGoldenOne/singlePlane_samePos/plane_160/2019-12-09_06.03.01/stack_2_channel_2/TP_00000/LC/Cam_Left_*.tif",
    "t0541c2p160",
    length=200,
    AffineTransform="from_x_path",
    dynamic_z_slice_mod=241,
    z_slices=[160],
)
t0541c2p170 = NamedDatasetInfo(
    Path("/"),
    "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.41.14_theGoldenOne/singlePlane_samePos/plane_170/2019-12-09_06.11.51/stack_2_channel_2/TP_00000/RC_rectified/Cam_Right_*_rectified.tif",
    "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.41.14_theGoldenOne/singlePlane_samePos/plane_170/2019-12-09_06.11.51/stack_2_channel_2/TP_00000/LC/Cam_Left_*.tif",
    "t0541c2p170",
    length=200,
    AffineTransform="from_x_path",
    dynamic_z_slice_mod=241,
    z_slices=[170],
)
t0541c2p180 = NamedDatasetInfo(
    Path("/"),
    "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.41.14_theGoldenOne/singlePlane_samePos/plane_180/2019-12-09_06.04.27/stack_2_channel_2/TP_00000/RC_rectified/Cam_Right_*_rectified.tif",
    "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.41.14_theGoldenOne/singlePlane_samePos/plane_180/2019-12-09_06.04.27/stack_2_channel_2/TP_00000/LC/Cam_Left_*.tif",
    "t0541c2p180",
    length=200,
    AffineTransform="from_x_path",
    dynamic_z_slice_mod=241,
    z_slices=[180],
)

t0541c3p30 = NamedDatasetInfo(
    Path("/"),
    "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.41.14_theGoldenOne/singlePlane_samePos/plane_030/2019-12-09_06.07.03/stack_2_channel_3/TP_00000/RC_rectified/Cam_Right_*_rectified.tif",
    "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.41.14_theGoldenOne/singlePlane_samePos/plane_030/2019-12-09_06.07.03/stack_2_channel_3/TP_00000/LC/Cam_Left_*.tif",
    "t0541c3p30",
    length=200,
    AffineTransform="from_x_path",
    dynamic_z_slice_mod=241,
    z_slices=[30],
)
t0541c3p40 = NamedDatasetInfo(
    Path("/"),
    "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.41.14_theGoldenOne/singlePlane_samePos/plane_040/2019-12-09_06.05.50/stack_2_channel_3/TP_00000/RC_rectified/Cam_Right_*_rectified.tif",
    "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.41.14_theGoldenOne/singlePlane_samePos/plane_040/2019-12-09_06.05.50/stack_2_channel_3/TP_00000/LC/Cam_Left_*.tif",
    "t0541c3p40",
    length=200,
    AffineTransform="from_x_path",
    dynamic_z_slice_mod=241,
    z_slices=[40],
)
t0541c3p50 = NamedDatasetInfo(
    Path("/"),
    "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.41.14_theGoldenOne/singlePlane_samePos/plane_050/2019-12-09_06.15.21/stack_2_channel_3/TP_00000/RC_rectified/Cam_Right_*_rectified.tif",
    "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.41.14_theGoldenOne/singlePlane_samePos/plane_050/2019-12-09_06.15.21/stack_2_channel_3/TP_00000/LC/Cam_Left_*.tif",
    "t0541c3p50",
    length=200,
    AffineTransform="from_x_path",
    dynamic_z_slice_mod=241,
    z_slices=[50],
)
t0541c3p60 = NamedDatasetInfo(
    Path("/"),
    "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.41.14_theGoldenOne/singlePlane_samePos/plane_060/2019-12-09_06.01.42/stack_2_channel_3/TP_00000/RC_rectified/Cam_Right_*_rectified.tif",
    "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.41.14_theGoldenOne/singlePlane_samePos/plane_060/2019-12-09_06.01.42/stack_2_channel_3/TP_00000/LC/Cam_Left_*.tif",
    "t0541c3p60",
    length=200,
    AffineTransform="from_x_path",
    dynamic_z_slice_mod=241,
    z_slices=[60],
)
t0541c3p70 = NamedDatasetInfo(
    Path("/"),
    "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.41.14_theGoldenOne/singlePlane_samePos/plane_070/2019-12-09_06.14.09/stack_2_channel_3/TP_00000/RC_rectified/Cam_Right_*_rectified.tif",
    "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.41.14_theGoldenOne/singlePlane_samePos/plane_070/2019-12-09_06.14.09/stack_2_channel_3/TP_00000/LC/Cam_Left_*.tif",
    "t0541c3p70",
    length=200,
    AffineTransform="from_x_path",
    dynamic_z_slice_mod=241,
    z_slices=[70],
)
t0541c3p80 = NamedDatasetInfo(
    Path("/"),
    "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.41.14_theGoldenOne/singlePlane_samePos/plane_080/2019-12-09_05.58.38/stack_2_channel_3/TP_00000/RC_rectified/Cam_Right_*_rectified.tif",
    "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.41.14_theGoldenOne/singlePlane_samePos/plane_080/2019-12-09_05.58.38/stack_2_channel_3/TP_00000/LC/Cam_Left_*.tif",
    "t0541c3p80",
    length=200,
    AffineTransform="from_x_path",
    dynamic_z_slice_mod=241,
    z_slices=[80],
)

t0541c3p90 = NamedDatasetInfo(
    Path("/"),
    "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.41.14_theGoldenOne/singlePlane_samePos/plane_090/2019-12-09_06.13.02/stack_2_channel_3/TP_00000/RC_rectified/Cam_Right_*_rectified.tif",
    "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.41.14_theGoldenOne/singlePlane_samePos/plane_090/2019-12-09_06.13.02/stack_2_channel_3/TP_00000/LC/Cam_Left_*.tif",
    "t0541c3p90",
    length=200,
    AffineTransform="from_x_path",
    dynamic_z_slice_mod=241,
    z_slices=[90],
)
t0541c3p100 = NamedDatasetInfo(
    Path("/"),
    "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.41.14_theGoldenOne/singlePlane_samePos/plane_100/2019-12-09_05.55.26/stack_2_channel_3/TP_00000/RC_rectified/Cam_Right_*_rectified.tif",
    "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.41.14_theGoldenOne/singlePlane_samePos/plane_100/2019-12-09_05.55.26/stack_2_channel_3/TP_00000/LC/Cam_Left_*.tif",
    "t0541c3p100",
    length=200,
    AffineTransform="from_x_path",
    dynamic_z_slice_mod=241,
    z_slices=[100],
)
t0541c3p110 = NamedDatasetInfo(
    Path("/"),
    "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.41.14_theGoldenOne/singlePlane_samePos/plane_110/2019-12-09_06.08.14/stack_2_channel_3/TP_00000/RC_rectified/Cam_Right_*_rectified.tif",
    "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.41.14_theGoldenOne/singlePlane_samePos/plane_110/2019-12-09_06.08.14/stack_2_channel_3/TP_00000/LC/Cam_Left_*.tif",
    "t0541c3p110",
    length=200,
    AffineTransform="from_x_path",
    dynamic_z_slice_mod=241,
    z_slices=[110],
)
t0541c3p120 = NamedDatasetInfo(
    Path("/"),
    "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.41.14_theGoldenOne/singlePlane_samePos/plane_120/2019-12-09_05.53.55/stack_2_channel_3/TP_00000/RC_rectified/Cam_Right_*_rectified.tif",
    "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.41.14_theGoldenOne/singlePlane_samePos/plane_120/2019-12-09_05.53.55/stack_2_channel_3/TP_00000/LC/Cam_Left_*.tif",
    "t0541c3p120",
    length=200,
    AffineTransform="from_x_path",
    dynamic_z_slice_mod=241,
    z_slices=[120],
)
t0541c3p130 = NamedDatasetInfo(
    Path("/"),
    "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.41.14_theGoldenOne/singlePlane_samePos/plane_130/2019-12-09_06.09.31/stack_2_channel_3/TP_00000/RC_rectified/Cam_Right_*_rectified.tif",
    "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.41.14_theGoldenOne/singlePlane_samePos/plane_130/2019-12-09_06.09.31/stack_2_channel_3/TP_00000/LC/Cam_Left_*.tif",
    "t0541c3p130",
    length=200,
    AffineTransform="from_x_path",
    dynamic_z_slice_mod=241,
    z_slices=[130],
)
t0541c3p140 = NamedDatasetInfo(
    Path("/"),
    "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.41.14_theGoldenOne/singlePlane_samePos/plane_140/2019-12-09_06.00.01/stack_2_channel_3/TP_00000/RC_rectified/Cam_Right_*_rectified.tif",
    "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.41.14_theGoldenOne/singlePlane_samePos/plane_140/2019-12-09_06.00.01/stack_2_channel_3/TP_00000/LC/Cam_Left_*.tif",
    "t0541c3p140",
    length=200,
    AffineTransform="from_x_path",
    dynamic_z_slice_mod=241,
    z_slices=[140],
)
t0541c3p150 = NamedDatasetInfo(
    Path("/"),
    "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.41.14_theGoldenOne/singlePlane_samePos/plane_150/2019-12-09_06.10.38/stack_2_channel_3/TP_00000/RC_rectified/Cam_Right_*_rectified.tif",
    "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.41.14_theGoldenOne/singlePlane_samePos/plane_150/2019-12-09_06.10.38/stack_2_channel_3/TP_00000/LC/Cam_Left_*.tif",
    "t0541c3p150",
    length=200,
    AffineTransform="from_x_path",
    dynamic_z_slice_mod=241,
    z_slices=[150],
)
t0541c3p160 = NamedDatasetInfo(
    Path("/"),
    "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.41.14_theGoldenOne/singlePlane_samePos/plane_160/2019-12-09_06.03.01/stack_2_channel_3/TP_00000/RC_rectified/Cam_Right_*_rectified.tif",
    "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.41.14_theGoldenOne/singlePlane_samePos/plane_160/2019-12-09_06.03.01/stack_2_channel_3/TP_00000/LC/Cam_Left_*.tif",
    "t0541c3p160",
    length=200,
    AffineTransform="from_x_path",
    dynamic_z_slice_mod=241,
    z_slices=[160],
)
t0541c3p170 = NamedDatasetInfo(
    Path("/"),
    "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.41.14_theGoldenOne/singlePlane_samePos/plane_170/2019-12-09_06.11.51/stack_2_channel_3/TP_00000/RC_rectified/Cam_Right_*_rectified.tif",
    "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.41.14_theGoldenOne/singlePlane_samePos/plane_170/2019-12-09_06.11.51/stack_2_channel_3/TP_00000/LC/Cam_Left_*.tif",
    "t0541c3p170",
    length=200,
    AffineTransform="from_x_path",
    dynamic_z_slice_mod=241,
    z_slices=[170],
)
t0541c3p180 = NamedDatasetInfo(
    Path("/"),
    "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.41.14_theGoldenOne/singlePlane_samePos/plane_180/2019-12-09_06.04.27/stack_2_channel_3/TP_00000/RC_rectified/Cam_Right_*_rectified.tif",
    "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.41.14_theGoldenOne/singlePlane_samePos/plane_180/2019-12-09_06.04.27/stack_2_channel_3/TP_00000/LC/Cam_Left_*.tif",
    "t0541c3p180",
    length=200,
    AffineTransform="from_x_path",
    dynamic_z_slice_mod=241,
    z_slices=[180],
)


t0402c10p80a = NamedDatasetInfo(
    Path("/"),
    "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/fast_cropped_8ms/singlePlanes/plane_080/2019-12-09_04.02.24/stack_2_channel_10/TP_00000/RC_rectified/Cam_Right_*_rectified.tif",
    "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/fast_cropped_8ms/singlePlanes/plane_080/2019-12-09_04.02.24/stack_2_channel_10/TP_00000/LC/Cam_Left_*.tif",
    "t0402c10p80a",
    length=200,
    AffineTransform="from_x_path",
    dynamic_z_slice_mod=241,
    z_slices=[80],
)
t0402c11p80a = NamedDatasetInfo(
    Path("/"),
    "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/fast_cropped_8ms/singlePlanes/plane_080/2019-12-09_04.02.24/stack_2_channel_11/TP_00000/RC_rectified/Cam_Right_*_rectified.tif",
    "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/fast_cropped_8ms/singlePlanes/plane_080/2019-12-09_04.02.24/stack_2_channel_11/TP_00000/LC/Cam_Left_*.tif",
    "t0402c11p80a",
    length=200,
    AffineTransform="from_x_path",
    dynamic_z_slice_mod=241,
    z_slices=[80],
)
t0402c10p80b = NamedDatasetInfo(
    Path("/"),
    "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/fast_cropped_8ms/singlePlanes/plane_080/2019-12-09_04.05.07_irisOpenedComplete/stack_2_channel_10/TP_00000/RC_rectified/Cam_Right_*_rectified.tif",
    "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/fast_cropped_8ms/singlePlanes/plane_080/2019-12-09_04.05.07_irisOpenedComplete/stack_2_channel_10/TP_00000/LC/Cam_Left_*.tif",
    "t0402c10p80b",
    length=200,
    AffineTransform="from_x_path",
    dynamic_z_slice_mod=241,
    z_slices=[80],
)
t0402c11p80b = NamedDatasetInfo(
    Path("/"),
    "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/fast_cropped_8ms/singlePlanes/plane_080/2019-12-09_04.05.07_irisOpenedComplete/stack_2_channel_11/TP_00000/RC_rectified/Cam_Right_*_rectified.tif",
    "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/fast_cropped_8ms/singlePlanes/plane_080/2019-12-09_04.05.07_irisOpenedComplete/stack_2_channel_11/TP_00000/LC/Cam_Left_*.tif",
    "t0402c11p80b",
    length=200,
    AffineTransform="from_x_path",
    dynamic_z_slice_mod=241,
    z_slices=[80],
)
t0402c10p100a = NamedDatasetInfo(
    Path("/"),
    "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/fast_cropped_8ms/singlePlanes/plane_100/2019-12-09_03.44.34/stack_2_channel_10/TP_00000/RC_rectified/Cam_Right_*_rectified.tif",
    "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/fast_cropped_8ms/singlePlanes/plane_100/2019-12-09_03.44.34/stack_2_channel_10/TP_00000/LC/Cam_Left_*.tif",
    "t0402c10p100a",
    length=200,
    AffineTransform="from_x_path",
    dynamic_z_slice_mod=241,
    z_slices=[100],
)
t0402c11p100a = NamedDatasetInfo(
    Path("/"),
    "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/fast_cropped_8ms/singlePlanes/plane_100/2019-12-09_03.44.34/stack_2_channel_11/TP_00000/RC_rectified/Cam_Right_*_rectified.tif",
    "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/fast_cropped_8ms/singlePlanes/plane_100/2019-12-09_03.44.34/stack_2_channel_11/TP_00000/LC/Cam_Left_*.tif",
    "t0402c11p100a",
    length=200,
    AffineTransform="from_x_path",
    dynamic_z_slice_mod=241,
    z_slices=[100],
)
t0402c10p100b = NamedDatasetInfo(
    Path("/"),
    "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/fast_cropped_8ms/singlePlanes/plane_100/2019-12-09_03.46.56/stack_2_channel_10/TP_00000/RC_rectified/Cam_Right_*_rectified.tif",
    "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/fast_cropped_8ms/singlePlanes/plane_100/2019-12-09_03.46.56/stack_2_channel_10/TP_00000/LC/Cam_Left_*.tif",
    "t0402c10p100b",
    length=200,
    AffineTransform="from_x_path",
    dynamic_z_slice_mod=241,
    z_slices=[100],
)
t0402c11p100b = NamedDatasetInfo(
    Path("/"),
    "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/fast_cropped_8ms/singlePlanes/plane_100/2019-12-09_03.46.56/stack_2_channel_11/TP_00000/RC_rectified/Cam_Right_*_rectified.tif",
    "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/fast_cropped_8ms/singlePlanes/plane_100/2019-12-09_03.46.56/stack_2_channel_11/TP_00000/LC/Cam_Left_*.tif",
    "t0402c11p100b",
    length=200,
    AffineTransform="from_x_path",
    dynamic_z_slice_mod=241,
    z_slices=[100],
)
t0402c10p120a = NamedDatasetInfo(
    Path("/"),
    "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/fast_cropped_8ms/singlePlanes/plane_120/2019-12-09_03.41.23/stack_2_channel_10/TP_00000/RC_rectified/Cam_Right_*_rectified.tif",
    "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/fast_cropped_8ms/singlePlanes/plane_120/2019-12-09_03.41.23/stack_2_channel_10/TP_00000/LC/Cam_Left_*.tif",
    "t0402c10p120a",
    length=200,
    AffineTransform="from_x_path",
    dynamic_z_slice_mod=241,
    z_slices=[120],
)
t0402c11p120a = NamedDatasetInfo(
    Path("/"),
    "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/fast_cropped_8ms/singlePlanes/plane_120/2019-12-09_03.41.23/stack_2_channel_11/TP_00000/RC_rectified/Cam_Right_*_rectified.tif",
    "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/fast_cropped_8ms/singlePlanes/plane_120/2019-12-09_03.41.23/stack_2_channel_11/TP_00000/LC/Cam_Left_*.tif",
    "t0402c11p120a",
    length=200,
    AffineTransform="from_x_path",
    dynamic_z_slice_mod=241,
    z_slices=[120],
)
t0402c10p120b = NamedDatasetInfo(
    Path("/"),
    "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/fast_cropped_8ms/singlePlanes/plane_120/2019-12-09_03.42.18/stack_2_channel_10/TP_00000/RC_rectified/Cam_Right_*_rectified.tif",
    "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/fast_cropped_8ms/singlePlanes/plane_120/2019-12-09_03.42.18/stack_2_channel_10/TP_00000/LC/Cam_Left_*.tif",
    "t0402c10p120b",
    length=200,
    AffineTransform="from_x_path",
    dynamic_z_slice_mod=241,
    z_slices=[120],
)
t0402c11p120b = NamedDatasetInfo(
    Path("/"),
    "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/fast_cropped_8ms/singlePlanes/plane_120/2019-12-09_03.42.18/stack_2_channel_11/TP_00000/RC_rectified/Cam_Right_*_rectified.tif",
    "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/fast_cropped_8ms/singlePlanes/plane_120/2019-12-09_03.42.18/stack_2_channel_11/TP_00000/LC/Cam_Left_*.tif",
    "t0402c11p120b",
    length=200,
    AffineTransform="from_x_path",
    dynamic_z_slice_mod=241,
    z_slices=[120],
)
t0402c10p140a = NamedDatasetInfo(
    Path("/"),
    "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/fast_cropped_8ms/singlePlanes/plane_140/2019-12-09_03.55.51/stack_2_channel_10/TP_00000/RC_rectified/Cam_Right_*_rectified.tif",
    "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/fast_cropped_8ms/singlePlanes/plane_140/2019-12-09_03.55.51/stack_2_channel_10/TP_00000/LC/Cam_Left_*.tif",
    "t0402c10p140a",
    length=200,
    AffineTransform="from_x_path",
    dynamic_z_slice_mod=241,
    z_slices=[140],
)
t0402c11p140a = NamedDatasetInfo(
    Path("/"),
    "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/fast_cropped_8ms/singlePlanes/plane_140/2019-12-09_03.55.51/stack_2_channel_11/TP_00000/RC_rectified/Cam_Right_*_rectified.tif",
    "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/fast_cropped_8ms/singlePlanes/plane_140/2019-12-09_03.55.51/stack_2_channel_11/TP_00000/LC/Cam_Left_*.tif",
    "t0402c11p140a",
    length=200,
    AffineTransform="from_x_path",
    dynamic_z_slice_mod=241,
    z_slices=[140],
)
t0402c10p140b = NamedDatasetInfo(
    Path("/"),
    "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/fast_cropped_8ms/singlePlanes/plane_140/2019-12-09_03.56.44/stack_2_channel_10/TP_00000/RC_rectified/Cam_Right_*_rectified.tif",
    "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/fast_cropped_8ms/singlePlanes/plane_140/2019-12-09_03.56.44/stack_2_channel_10/TP_00000/LC/Cam_Left_*.tif",
    "t0402c10p140b",
    length=200,
    AffineTransform="from_x_path",
    dynamic_z_slice_mod=241,
    z_slices=[140],
)
t0402c11p140b = NamedDatasetInfo(
    Path("/"),
    "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/fast_cropped_8ms/singlePlanes/plane_140/2019-12-09_03.56.44/stack_2_channel_11/TP_00000/RC_rectified/Cam_Right_*_rectified.tif",
    "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/fast_cropped_8ms/singlePlanes/plane_140/2019-12-09_03.56.44/stack_2_channel_11/TP_00000/LC/Cam_Left_*.tif",
    "t0402c11p140b",
    length=200,
    AffineTransform="from_x_path",
    dynamic_z_slice_mod=241,
    z_slices=[140],
)
t0402c10p160a = NamedDatasetInfo(
    Path("/"),
    "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/fast_cropped_8ms/singlePlanes/plane_160/2019-12-09_03.58.24/stack_2_channel_10/TP_00000/RC_rectified/Cam_Right_*_rectified.tif",
    "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/fast_cropped_8ms/singlePlanes/plane_160/2019-12-09_03.58.24/stack_2_channel_10/TP_00000/LC/Cam_Left_*.tif",
    "t0402c10p160a",
    length=200,
    AffineTransform="from_x_path",
    dynamic_z_slice_mod=241,
    z_slices=[160],
)
t0402c11p160a = NamedDatasetInfo(
    Path("/"),
    "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/fast_cropped_8ms/singlePlanes/plane_160/2019-12-09_03.58.24/stack_2_channel_11/TP_00000/RC_rectified/Cam_Right_*_rectified.tif",
    "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/fast_cropped_8ms/singlePlanes/plane_160/2019-12-09_03.58.24/stack_2_channel_11/TP_00000/LC/Cam_Left_*.tif",
    "t0402c11p160a",
    length=200,
    AffineTransform="from_x_path",
    dynamic_z_slice_mod=241,
    z_slices=[160],
)
t0402c10p160b = NamedDatasetInfo(
    Path("/"),
    "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/fast_cropped_8ms/singlePlanes/plane_160/2019-12-09_03.59.45/stack_2_channel_10/TP_00000/RC_rectified/Cam_Right_*_rectified.tif",
    "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/fast_cropped_8ms/singlePlanes/plane_160/2019-12-09_03.59.45/stack_2_channel_10/TP_00000/LC/Cam_Left_*.tif",
    "t0402c10p160b",
    length=200,
    AffineTransform="from_x_path",
    dynamic_z_slice_mod=241,
    z_slices=[160],
)
t0402c11p160b = NamedDatasetInfo(
    Path("/"),
    "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/fast_cropped_8ms/singlePlanes/plane_160/2019-12-09_03.59.45/stack_2_channel_11/TP_00000/RC_rectified/Cam_Right_*_rectified.tif",
    "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/fast_cropped_8ms/singlePlanes/plane_160/2019-12-09_03.59.45/stack_2_channel_11/TP_00000/LC/Cam_Left_*.tif",
    "t0402c11p160b",
    length=200,
    AffineTransform="from_x_path",
    dynamic_z_slice_mod=241,
    z_slices=[160],
)
