from hylfm.datasets import get_tensor_info
from hylfm.datasets.base import H5Dataset


def test_get_from_zenodo():
    info = get_tensor_info("beads.small_0", "lf", {})
    assert info.download_file_path.exists()
    assert info.download_file_path.is_file()
    assert info.root.exists()
    assert info.root.is_dir()

def test_get_h5_from_zenodo():
    info = get_tensor_info("beads.small_2", "ls_reg", {})
    assert info.download_file_path.exists()
    assert info.download_file_path.is_file()
    assert info.root.exists()
    ds = H5Dataset(info=info)
    assert len(ds) == 137
    tensors = ds[0]
    assert "ls_reg" in tensors
    img = tensors["ls_reg"]
    assert img.shape == (1, 1, 121, 392, 592)

