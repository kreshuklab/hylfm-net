from pathlib import Path

import pytest
from lnet.datasets import get_dataset_from_info, N5CachedDatasetFromInfo

from lnet.datasets.gcamp import g200311_083021_ls


@pytest.fixture
def data_path() -> Path:
    return Path(__file__).parent / "testdata"


@pytest.fixture
def dummy_config_path(data_path: Path) -> Path:
    return data_path / "experiment_configs/dummy.yml"


@pytest.fixture
def test_ls_slice_dataset() -> N5CachedDatasetFromInfo:
    info = g200311_083021_ls
    info.transformations += [
        {
            "Resize": {
                "apply_to": "ls",
                "shape": [1.0, 1.0, 0.21052631578947368421052631578947, 0.21052631578947368421052631578947],
                "order": 2,
            }
        },  # 2/19=0.10526315789473684210526315789474; 4/19=0.21052631578947368421052631578947; 8/19=0.42105263157894736842105263157895
        {"Assert": {"apply_to": "ls", "expected_tensor_shape": [None, 1, 1, None, None]}},
    ]
    return get_dataset_from_info(info=info, cache=True)


# @pytest.fixture
# def test_ls_subdataset(test_ls_dataset):
#     return N5Sub
