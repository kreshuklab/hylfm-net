import shutil

from lnet.settings import settings

root = settings.log_path

delete = """
/g/kreshuk/LF_computed/lnet/logs/beads/z_out51/01highc_f4/20-04-29_16-24-00
/g/kreshuk/LF_computed/lnet/logs/beads/z_out51/4mu_f4/20-04-29_16-25-14
/g/kreshuk/LF_computed/lnet/logs/beads/z_out51/01highc_f8/20-04-29_16-24-44
/g/kreshuk/LF_computed/lnet/logs/beads/z_out51/4mu_f8/20-04-29_16-25-43
"""

for name in delete.strip("\n").split("\n"):
    shutil.rmtree(root / name)
    # for file_path in data_path.glob(f"{name}.*"):
    #     file_path.unlink()
