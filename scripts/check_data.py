import logging
import re
from pathlib import Path

from imageio import volread

logger = logging.getLogger(__name__)


# path = Path(
#     "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_04.54.38/stack_1_channel_2"
# )
path = Path(
    "/g/hufnagel/LF/LenseLeNet_Microscope/20191030_Beads_massiveGT/Beads_1micron/2019-10-30_05.20.28_withSimultIllum/"
    "stack_1_channel_1"
)

assert path.exists(), path.absolute()
# LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_04.54.38/stack_1_channel_2
tp_glob = "TP_*"
# glob_expr = "LC/Cam_Left_*.tif"
# glob_expr = "RC_rectified/Cam_Right_*_rectified.tif"
glob_expr = "RC_rectified/Cam_Right_1_rectified.tif"
glob_expr = "LC/Cam_Left_registered.tif"

expected = [(n,) for n in range(1, 242)]

count = 0
for i, tp in enumerate(path.glob(tp_glob)):
    found_paths = list(tp.glob(glob_expr))
#     glob_numbers = [nr for nr in re.findall(r"\d+", glob_expr)]
#     logger.info("found numbers %s in glob_exp %s", glob_numbers, glob_expr)
#     numbers = [
#         tuple(int(nr) for nr in re.findall(r"\d+", p.relative_to(tp).as_posix()) if nr not in glob_numbers)
#         for p in found_paths
#     ]
#     count += len(numbers)
#     print(i, count, [(e, f) for e, f in zip(expected, numbers) if e != f])
    
    print(volread(found_paths[0]).shape)
