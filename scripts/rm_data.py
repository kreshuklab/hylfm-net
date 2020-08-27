import shutil
from pathlib import Path

from hylfm.settings import settings

# root = settings.log_dir

# delete = """
# /g/kreshuk/LF_computed/lnet/logs/beads/z_out51/01highc_f4/20-04-29_16-24-00
# /g/kreshuk/LF_computed/lnet/logs/beads/z_out51/4mu_f4/20-04-29_16-25-14
# /g/kreshuk/LF_computed/lnet/logs/beads/z_out51/01highc_f8/20-04-29_16-24-44
# /g/kreshuk/LF_computed/lnet/logs/beads/z_out51/4mu_f8/20-04-29_16-25-43
# """

# for name in delete.strip("\n").split("\n"):
#     shutil.rmtree(root / name)
#     # for file_path in data_path.glob(f"{name}.*"):
#     #     file_path.unlink()

# rm_paths = list(settings.cache_dir.glob("*mask.npy"))
# for rm_path in rm_paths:
#     print(rm_path)
#     try:
#         shutil.rmtree(rm_path)
#     except NotADirectoryError:
#         rm_path.unlink()

    # txt = txt_path.read_text()
    # if "index_mask" not in str(txt_path) and "quality" in txt:
    #     print(txt)
    #     print(txt_path.with_suffix(""))
    #     rm_paths = list(settings.cache_dir.glob((txt_path.with_suffix("").name + "*")))
    #     if len(rm_paths) == 1:
    #         rm_path = rm_paths[0]
    #         print(rm_path)
    #
    #         try:
    #             shutil.rmtree(rm_path)
    #         except NotADirectoryError:
    #             rm_path.unlink()
    #     else:
    #         print(rm_paths)
    #         break

# for name in delete.strip("\n").split("\n"):
#     txt =
#     # shutil.rmtree(root / name)
#     # for file_path in data_path.glob(f"{name}.*"):
#     #     file_path.unlink()

if __name__ =="__main__":
    root = Path("/scratch/beuttenm/lnet/data4/20191208_234342_ls_slice_61bfbb5e0699f177e586248ba6341b845d04d0b593fcdab495916a81.n5/ls_slice/0/0/0/0")
    sizes = {}
    for file in root.glob("*"):
        size = file.stat().st_size
        count = sizes.get(size, 0) + 1
        sizes[size] = count
        if size not in [69565, 67071]:
            print(size, file)
            break

    print(sizes)
    print(max(sizes.values()))
