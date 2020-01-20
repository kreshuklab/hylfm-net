from pathlib import Path
from typing import List, Optional

import torch
from scipy.ndimage import zoom
from torch.utils.data import DataLoader

from lnet.metrics import MSSSIM, SSIM  # SSIMSkImage != SSIM
from lnet.metrics import NRMSE
from lnet.metrics import PSNR

from tifffile import imread
from torch.utils.data.dataset import Dataset


class FileListDataset(Dataset):
    def __init__(self, file_paths: List[Path], z_out: Optional[int]):
        assert all([fp.exists() for fp in file_paths])
        self.file_paths = file_paths
        self.z_out = z_out

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, item):
        img = imread(self.file_paths[item].as_posix()).astype("float32")
        if self.z_out is None or img.shape[0] == self.z_out:
            return img[None, None, ...]
        else:
            z_zoom = self.z_out / img.shape[0]
            return zoom(img, (z_zoom, 1, 1), order=1)[None, None, ...]


def get_file_paths(glob_path: str):
    path = glob_path[: glob_path.find("*")]
    sep_idx = path.rfind("/")
    path = Path(path[:sep_idx])
    assert path.exists()
    glob_expr = glob_path[sep_idx + 1 :]
    return sorted(list(path.glob(glob_expr)))


def evaluate(ipt_glob_path: str, tgt_glob_path: str, z_out: Optional[int] = None):
    metrics = [MSSSIM(), SSIM(), NRMSE(), PSNR()]
    ipts = get_file_paths(ipt_glob_path)
    tgts = get_file_paths(tgt_glob_path)
    assert len(ipts) == len(tgts)
    ipts = DataLoader(FileListDataset(ipts, z_out=z_out), batch_size=1, shuffle=False, num_workers=8)
    tgts = DataLoader(FileListDataset(tgts, z_out=z_out), batch_size=1, shuffle=False, num_workers=8)

    for ipt, tgt in zip(ipts, tgts):
        [m.update((ipt, tgt)) for m in metrics]

    return dict(zip([m.__class__.__name__ for m in metrics], [m.compute() for m in metrics]))


if __name__ == "__main__":
    print(
        "LS-LS",
        evaluate(
            "/g/kreshuk/beuttenm/repos/lnet/logs/fish/test/20-01-20_13-22-09/test_data/target/fish1_20191208_0216_static/*.tif",
            "/g/kreshuk/beuttenm/repos/lnet/logs/fish/test/20-01-20_13-22-09/test_data/target/fish1_20191208_0216_static/*.tif",
        ),
    )
    # print([0.0])

    # print("full LR-LS", evaluate(
    #     "/g/kreshuk/LF_computed/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish1/static/Heart_tightCrop/centered_5steps_stepsize8/2019-12-09_02.16.30/stack_3_channel_0/TP_*/RCout/Cam_Right_001.tif",
    #     "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish1/static/Heart_tightCrop/centered_5steps_stepsize8/2019-12-09_02.16.30/stack_4_channel_1/TP_*/LC/Cam_Left_registered.tif",
    #     z_out=49,
    # ))
    # print([0.944491050046173])

    print(
        "LR-LS",
        evaluate(
            "/g/kreshuk/beuttenm/repos/lnet/logs/fish/test/20-01-20_13-22-09/test_data/target/fish1_20191208_0216_static_lr/*.tif",
            "/g/kreshuk/beuttenm/repos/lnet/logs/fish/test/20-01-20_13-22-09/test_data/target/fish1_20191208_0216_static/*.tif",
        ),
    )
    # print([0.9044772294017559])

    print(
        "NN-LS",
        evaluate(
            "/g/kreshuk/beuttenm/repos/lnet/logs/fish/test/20-01-20_13-22-09/test_data/prediction/fish1_20191208_0216_static/*.tif",
            "/g/kreshuk/beuttenm/repos/lnet/logs/fish/test/20-01-20_13-22-09/test_data/target/fish1_20191208_0216_static/*.tif",
        ),
    )

    print(
        "NN-LR",
        evaluate(
            "/g/kreshuk/beuttenm/repos/lnet/logs/fish/test/20-01-20_13-22-09/test_data/prediction/fish1_20191208_0216_static/*.tif",
            "/g/kreshuk/beuttenm/repos/lnet/logs/fish/test/20-01-20_13-22-09/test_data/target/fish1_20191208_0216_static_lr/*.tif",
        ),
    )
    # print([0.8583414594192545])


"""
    print("LS-LS", evaluate(
        "/g/kreshuk/beuttenm/repos/lnet/logs/fish/test/20-01-20_13-22-09/test_data/target/fish1_20191208_0216_static/*.tif",
        "/g/kreshuk/beuttenm/repos/lnet/logs/fish/test/20-01-20_13-22-09/test_data/target/fish1_20191208_0216_static/*.tif",
    ))
    # print([0.0])

    # print("full LR-LS", evaluate(
    #     "/g/kreshuk/LF_computed/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish1/static/Heart_tightCrop/centered_5steps_stepsize8/2019-12-09_02.16.30/stack_3_channel_0/TP_*/RCout/Cam_Right_001.tif",
    #     "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish1/static/Heart_tightCrop/centered_5steps_stepsize8/2019-12-09_02.16.30/stack_4_channel_1/TP_*/LC/Cam_Left_registered.tif",
    #     z_out=49,
    # ))
    # print([0.944491050046173])

    print("LR-LS", evaluate(
        "/g/kreshuk/beuttenm/repos/lnet/logs/fish/test/20-01-20_13-22-09/test_data/target/fish1_20191208_0216_static_lr/*.tif",
        "/g/kreshuk/beuttenm/repos/lnet/logs/fish/test/20-01-20_13-22-09/test_data/target/fish1_20191208_0216_static/*.tif",
    ))
    # print([0.9044772294017559])


    print("NN-LS", evaluate(
        "/g/kreshuk/beuttenm/repos/lnet/logs/fish/test/20-01-20_13-22-09/test_data/prediction/fish1_20191208_0216_static/*.tif",
        "/g/kreshuk/beuttenm/repos/lnet/logs/fish/test/20-01-20_13-22-09/test_data/target/fish1_20191208_0216_static/*.tif",
    ))

    print("NN-LR", evaluate(
        "/g/kreshuk/beuttenm/repos/lnet/logs/fish/test/20-01-20_13-22-09/test_data/prediction/fish1_20191208_0216_static/*.tif",
        "/g/kreshuk/beuttenm/repos/lnet/logs/fish/test/20-01-20_13-22-09/test_data/target/fish1_20191208_0216_static_lr/*.tif",
    ))
    # print([0.8583414594192545])
"""
