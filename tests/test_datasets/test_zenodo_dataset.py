from hylfm.datasets import get_tensor_info


def test_get_from_zenodo(test_zenodo_doi):
    info = get_tensor_info("beads.small_0", "lf", {})
    assert info.download_file_path.exists()
    assert info.download_file_path.is_file()
    assert info.root.exists()
    assert info.root.is_dir()
