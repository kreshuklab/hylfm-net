from hylfm.datasets import N5CachedDatasetFromInfoSubset


def test_intensity_range(ls_slice_dataset):
    ds = N5CachedDatasetFromInfoSubset(
        dataset=ls_slice_dataset,
        indices=[0, 1],
        filters=[("instensity_range", {"apply_to": "ls", "max_above": {"percentile": 99.99}})],
    )

    assert len(ds) == 1, len(ds)
