from lnet.datasets import N5CachedDatasetFromInfoSubset


def test_intensity_range(test_ls_slice_dataset):
    ds = N5CachedDatasetFromInfoSubset(dataset=test_ls_slice_dataset, indices=[0, 40], filters=[("instensity_range", {"apply_to": "ls", "max_above": {"percentile": 20}})])

    assert len(ds) == 1, len(ds)
