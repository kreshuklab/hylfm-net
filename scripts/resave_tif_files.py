import argparse
import shutil
from pathlib import Path

import tifffile
import yaml


def volread(p: Path):
    with p.open("rb") as f:
        return tifffile.imread(f)


def volwrite(p: Path, data, compress=2, **kwargs):
    with p.open("wb") as f:
        tifffile.imsave(f, data, compress=compress, **kwargs)


def resave_files(file_glob: str, out_path: Path):
    out_path.mkdir(exist_ok=True)

    root = Path(file_glob.split("*")[0]).parent
    print("root", root)
    glob_expr = str(Path(file_glob).relative_to(root))
    print("glob_expr", glob_expr)

    paths = list(root.glob(glob_expr))
    all_path_parts = [p.relative_to(root).parts for p in paths]
    nparts = len(all_path_parts[0])
    assert all(len(pp) == nparts for pp in all_path_parts)
    diff_parts = [set(pp[i] for pp in all_path_parts) for i in range(nparts)]
    part_indices_for_name = [i for i, dp in enumerate(diff_parts) if len(dp) > 1]

    def get_name(path_parts):
        return "_".join([path_parts[i] for i in part_indices_for_name])

    print("example name", get_name(all_path_parts[0]))

    for src, parts in zip(paths, all_path_parts):
        dest = (out_path / get_name(parts)).with_suffix(src.suffix)
        if not dest.exists():
            img = volread(src)
            volwrite(dest, img)


data_roots = {
    "GHUFNAGELLFLenseLeNet_Microscope": Path("Y:/"),
    # "GKRESHUK": Path("K:/"),
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # parser.add_argument("file_glob")
    # parser.add_argument("out_path", type=Path)

    # file_glob: str = args.file_glob
    # out_path: Path = args.out_path
    # copy_files(file_glob, out_path)

    parser.add_argument("yml_config", type=Path)

    args = parser.parse_args()
    with args.yml_config.open() as f:
        config = yaml.safe_load(f)

    for out, dat in config.items():
        resave_files(str(Path(data_roots[dat["root"]]) / dat["location"]), Path(out))
        shutil.make_archive(out, 'zip', out)

