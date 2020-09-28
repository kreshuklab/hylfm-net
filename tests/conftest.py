from collections import OrderedDict
from pathlib import Path

import pytest
import typing
from ruamel.yaml import YAML

from hylfm import settings
from hylfm.datasets import N5CachedDatasetFromInfo, ZipDataset, get_dataset_from_info, get_tensor_info

yaml = YAML(typ="safe")


@pytest.fixture
def output_path() -> Path:
    output_path = Path(__file__).parent / "output_test_data"
    output_path.mkdir(exist_ok=True)
    return output_path


@pytest.fixture
def dummy_config_path() -> Path:
    return settings.configs_dir / "dummy.yml"


@pytest.fixture
def meta() -> dict:
    return {"nnum": 19, "z_out": 51, "interpolation_order": 2, "scale": 2}


# @pytest.fixture
# def ls_slice_dataset(meta) -> N5CachedDatasetFromInfo:
#     info = get_tensor_info("brain.11_1__2020-03-11_03.22.33__SinglePlane_-330", "ls_slice", meta=meta)
#     return ZipDataset({"ls_slice": get_dataset_from_info(info=info, cache=True)})


@pytest.fixture
def beads_dataset(meta) -> N5CachedDatasetFromInfo:
    datasets = OrderedDict()
    for name in ["lf", "ls_reg"]:
        info = get_tensor_info("beads.small1", name, meta=meta)
        datasets[name] = get_dataset_from_info(info=info, cache=True)

    return ZipDataset(datasets)
