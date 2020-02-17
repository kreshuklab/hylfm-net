from pathlib import Path

path = Path("/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_04.54.38/stack_1_channel_3")
tp_glob = "TP_*"
# glob = "LC/Cam_Left_*.tif"
glob = "RC_rectified/Cam_Right_*_rectified.tif"

for tp in path.glob(tp_glob):
    print(len(list(tp.glob(glob))))

