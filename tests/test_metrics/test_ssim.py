import torch

from lnet.metrics import MSSSIM, MSSSIM_Skimage


def test_mssim(beads_dataset):
    ls_trf = beads_dataset[0]["ls_trf"]
    assert len(ls_trf.shape) == 5, ls_trf.shape
    lr = beads_dataset[0]["ls_reg"]
    assert ls_trf.shape == lr.shape, (ls_trf.shape, lr.shape)

    ls_trf = torch.from_numpy(ls_trf)
    lr = torch.from_numpy(lr)

    kwargs = {"window_size": 11, "size_average": True, "val_range": None, "normalize": False}
    torch_metric = MSSSIM(**kwargs)
    skimage_metric = MSSSIM_Skimage(**kwargs)

    torch_metric.update((lr, ls_trf))
    skimage_metric.update((lr, ls_trf))

    torch_mssim = torch_metric.compute()
    skimage_mssim = skimage_metric.compute()

    assert torch_mssim == skimage_mssim, (torch_mssim, skimage_mssim)
