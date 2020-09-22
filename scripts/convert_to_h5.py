import argparse

from z5py.converter import convert_to_h5

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("n5_in_path")
    parser.add_argument("h5_out_path")
    parser.add_argument("in_path_in_file")
    parser.add_argument("out_path_in_file")
    parser.add_argument("--n_threads", type=int, default=16)

    args = parser.parse_args()

    convert_to_h5(
        args.n5_in_path, args.h5_out_path, args.in_path_in_file, args.out_path_in_file, n_threads=args.n_threads
    )

    """
    /scratch/beuttenm/hylfm/cache/b01highc_2_ls_reg_8743bf3e67c30c171928d3cc30e75a895047dae013848c7585034fa0.n5 /g/kreshuk/LF_computed/zenodo_upload/small_2.h5 ls_reg ls_reg
    """
