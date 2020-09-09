import logging
import re
from pathlib import Path
from typing import Tuple

logger = logging.getLogger(__name__)


def split_off_glob(path: Path) -> Tuple[Path, str]:
    not_glob = path.as_posix().split("*")[0]
    valid_path = Path(not_glob[: not_glob.rfind("/")])
    glob_str = path.relative_to(valid_path).as_posix()
    return valid_path, glob_str


def get_paths(location: Path):
    if "*" in str(location):
        folder, glob_expr = split_off_glob(location)
        logger.debug("split data location into %s and %s", folder, glob_expr)
        assert folder.exists(), folder.absolute()
        found_paths = list(folder.glob(glob_expr))
        glob_numbers = [nr for nr in re.findall(r"\d+", glob_expr)]
        logger.debug("found %d numbers in glob_exp %s", len(glob_numbers), glob_expr)
        numbers = [
            tuple(int(nr) for nr in re.findall(r"\d+", p.relative_to(folder).as_posix()) if nr not in glob_numbers)
            for p in found_paths
        ]
        logger.debug("found %d number tuples in folder %s", len(numbers), folder)
        # todo: check numbers for completeness
    else:
        assert location.exists(), location.absolute()
        found_paths = [location]

    length = len(str(found_paths[0]))
    if not all([len(str(p)) == length for p in found_paths]):
        raise ValueError("not all found paths have same lenght. make sure numbers are zero-padded.")

    return sorted(found_paths)


# regular_exp_h5 = "TestOutputGcamp(.*)SinglePlane_-[0-9]{3}"
# regular_exp_TP = "TP_([0-9]{5})"
#
#
#
# def gcamp_extract_LS_pos_from_path_of_rectified_file(path_to_rectified_file: Path):
#     assert path_to_rectified_file.exists()
#     LS_plane_pos_string = re.search(regular_exp_LS_pos, path_to_rectified_file.as_posix())
#     LS_plane_pos = int(LS_plane_pos_string.group(1))
#     assert LS_plane_pos is not None, (path_to_rectified_file, regular_exp_LS_pos)
#     return LS_plane_pos
#
#
# def gcamp_get_path_of_corresponding_LS_h5_file_from_rectified_file(path_to_rectified_file: Path, base_folder_h5: Path):
#     assert base_folder_h5.exists(), base_folder_h5
#     subpath_to_add_to_h5_parent_dir = re.search(regular_exp_h5, path_to_rectified_file.as_posix()).group(1)
#     TP_used = re.search(regular_exp_TP, path_to_rectified_file.as_posix()).group(1)
#     h5_file = base_folder_h5 / subpath_to_add_to_h5_parent_dir / f"Cam_Left_{TP_used}.h5"
#     assert h5_file.exists(), h5_file
#     return h5_file
#
#
# def gcamp_convert_LS_pos_in_original_stack_to_network_slice_pos(LS_stack_range_start, LS_plane_pos):
#     LS_pos_in_stack = abs(LS_stack_range_start - LS_plane_pos)
#     return LS_pos_in_stack


# #Check:
#
# path_to_corresponding_h5_file = get_path_of_corresponding_LS_h5_file_from_rectified_file(path_to_rectified_file, regular_exp_h5, base_folder_h5, regular_exp_TP)
#
# LS_plane_pos = extract_LS_pos_from_path_of_rectified_file(path_to_rectified_file, regular_exp_LS_pos)
#
# LS_pos_in_stack = convert_LS_pos_in_original_stack_to_network_slice_pos(LS_stack_range_start, LS_plane_pos)


# def get_image_paths(x_folder: Path, x_glob: str, y_folder: Optional[Path], y_glob: Optional[str], *, z_crop: Optional[Tuple[int, int]], z_min: Optional[int], z_max: Optional[int], z_dim: Optional[int], dynamic_z_slice_mod: Optional[int]):
#     if z_crop is None:
#         assert z_min is not None
#         assert z_max is not None
#         assert z_dim is not None
#     raw_x_paths, x_numbers = get_paths_and_numbers(x_folder, x_glob)
#     common_numbers = set(x_numbers)
#
#     if y_folder is None:
#         raw_y_paths = []
#         y_paths = []
#     else:
#         raw_y_paths, y_numbers = get_paths_and_numbers(y_folder, y_glob)
#         common_numbers &= set(y_numbers)
#         assert len(common_numbers) > 1 or len(set(x_numbers) | set(y_numbers)) == 1, (
#             "x",
#             set(x_numbers),
#             "y",
#             set(y_numbers),
#             "x|y",
#             set(x_numbers) | set(y_numbers),
#             "x&y",
#             set(x_numbers) & set(y_numbers),
#         )
#         y_paths = sorted([p for p, yn in zip(raw_y_paths, y_numbers) if yn in common_numbers])
#         y_drop = sorted([yn for yn in y_numbers if yn not in common_numbers])
#         logger.warning("dropping y: %s", y_drop)
#
#         if z_crop is not None:
#             y_paths = [
#                 p for i, p in enumerate(y_paths) if z_min <= i % dynamic_z_slice_mod <= z_max
#             ]
#
#     x_paths = sorted([p for p, xn in zip(raw_x_paths, x_numbers) if xn in common_numbers])
#     x_drop = sorted([xn for xn in x_numbers if xn not in common_numbers])
#     logger.warning("dropping x: %s", x_drop)
#     assert len(x_paths) >= 1, raw_x_paths
#
#     if z_crop is not None:
#         x_paths = [
#             p
#             for i, p in enumerate(x_paths)
#             if z_min <= i % dynamic_z_slice_mod and i % dynamic_z_slice_mod <= z_max
#         ]
#
#     if y_folder is not None:
#         assert len(x_paths) == len(y_paths), raw_y_paths
#
#     return x_paths, y_paths
