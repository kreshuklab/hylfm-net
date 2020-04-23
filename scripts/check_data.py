import logging
import os
import re
from pathlib import Path

from imageio import volread

logger = logging.getLogger(__name__)




# path = Path("/g/kreshuk/LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish2/10Hz/241Planes/2020-03-11_08.52.33/stack_1_channel_3/SwipeThrough_-450_-210_nimages_241")
path = Path("/g/kreshuk/LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish2/10Hz/241Planes/2020-03-11_09.08.00/stack_1_channel_3/SwipeThrough_-450_-210_nimages_241")
# path = Path("/g/kreshuk/LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish2/10Hz/slideThrough/2020-03-11_08.30.21/stack_36_channel_3/SwipeThrough_-390_-270_nimages_121")
# path = Path("/g/kreshuk/LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish2/10Hz/slideThrough/2020-03-11_08.34.19/stack_36_channel_3/SwipeThrough_-390_-270_nimages_121")
# path = Path("/g/kreshuk/LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish2/10Hz/slideThrough/2020-03-11_08.36.58/stack_36_channel_3/SwipeThrough_-390_-270_nimages_121")
# path = Path("/g/kreshuk/LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish2/10Hz/slideThrough/2020-03-11_08.39.31/stack_36_channel_3/SwipeThrough_-390_-270_nimages_121")
# path = Path("/g/kreshuk/LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish2/10Hz/slideThrough/2020-03-11_08.42.07/stack_36_channel_3/SwipeThrough_-390_-270_nimages_121")
# path = Path("/g/kreshuk/LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish2/10Hz/slideThrough/2020-03-11_08.44.50/stack_36_channel_3/SwipeThrough_-390_-270_nimages_121")
# path = Path("/g/kreshuk/LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/fish2/10Hz/slideThrough/2020-03-11_08.47.29/stack_36_channel_3/SwipeThrough_-390_-270_nimages_121")

for tp in path.glob("TP*"):
    # for i, cr in enumerate(tp.glob("RC_rectified/Cam_Right_*_rectified.tif")):
    # for i, cl in enumerate(tp.glob("LC/Cam_Left_*.tif")):
    #     if i > 180:
    #         cl.rename(cl.with_name(f"Cam_Left_{i+1}.tif.missing_rc_rectified"))
    #         print(i, cl.name)

    print(tp.name, len(list(tp.glob("RC_rectified/Cam_Right*.tif"))), len(list(tp.glob("LC/Cam_Left*.tif"))))
# # path = Path(
# #     "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_04.54.38/stack_1_channel_2"
# # )
# path = Path(
#     "/g/hufnagel/LF/LenseLeNet_Microscope/20191030_Beads_massiveGT/Beads_1micron/2019-10-30_05.20.28_withSimultIllum/"
#     "stack_1_channel_1"
# )
#
# ls TP_000$TP/RC_rectified/Cam_Right*.tif | wc && ls TP_000$TP/LC/Cam_Left*.tif | wc
#
# assert path.exists(), path.absolute()
# # LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_04.54.38/stack_1_channel_2
# tp_glob = "TP_*"
# # glob_expr = "LC/Cam_Left_*.tif"
# # glob_expr = "RC_rectified/Cam_Right_*_rectified.tif"
# glob_expr = "RC_rectified/Cam_Right_1_rectified.tif"
# glob_expr = "LC/Cam_Left_registered.tif"
#
# expected = [(n,) for n in range(1, 242)]
#
# count = 0
# for i, tp in enumerate(path.glob(tp_glob)):
#     found_paths = list(tp.glob(glob_expr))
#     len(found_paths)
# #     glob_numbers = [nr for nr in re.findall(r"\d+", glob_expr)]
# #     logger.info("found numbers %s in glob_exp %s", glob_numbers, glob_expr)
# #     numbers = [
# #         tuple(int(nr) for nr in re.findall(r"\d+", p.relative_to(tp).as_posix()) if nr not in glob_numbers)
# #         for p in found_paths
# #     ]
# #     count += len(numbers)
# #     print(i, count, [(e, f) for e, f in zip(expected, numbers) if e != f])
#
#     # print(volread(found_paths[0]).shape)
