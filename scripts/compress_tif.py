import argparse
from pathlib import Path

from tifffile import imsave, imread


def compress_tif(path: Path, glob: str, compress: int, for_real: bool):
    assert path.exists()

    for tif in path.glob(glob):
        try:
            if for_real:
                img = imread(str(tif))

            out_tif = tif.with_name(tif.name.replace(".tif", f"_compr{compress}.tif"))
            if for_real:
                imsave(str(out_tif), img, compress=compress, bigtiff=True)
                out_tif.rename(tif)
                # tif.unlink()
            else:
                print(tif)
        except Exception as e:
            print(tif, e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("glob", type=str)
    parser.add_argument("compress", type=int)
    parser.add_argument("--for-real", action="store_true")

    args = parser.parse_args()
    if args.for_real:
        input("for real?")

    compress_tif(path=args.path, glob=args.glob, compress=args.compress, for_real=args.for_real)
