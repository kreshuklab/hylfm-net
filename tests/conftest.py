from collections import OrderedDict
from pathlib import Path

import pytest
import typing
from ruamel.yaml import YAML

from hylfm.datasets import N5CachedDatasetFromInfo, ZipDataset, get_dataset_from_info, get_tensor_info

yaml = YAML(typ="safe")


@pytest.fixture
def data_path() -> Path:
    return Path(__file__).parent / "testdata"


@pytest.fixture
def output_path() -> Path:
    output_path = Path(__file__).parent / "output_test_data"
    output_path.mkdir(exist_ok=True)
    return output_path


@pytest.fixture
def dummy_config_path(data_path: Path) -> Path:
    return data_path / "experiment_configs/dummy.yml"


@pytest.fixture
def meta() -> dict:
    return {"nnum": 19, "z_out": 49, "interpolation_order": 2, "scale": 2}


@pytest.fixture
def ls_slice_dataset(meta) -> N5CachedDatasetFromInfo:
    info = get_tensor_info("brain.11_1__2020-03-11_03.22.33__SinglePlane_-330", "ls_slice", meta=meta)
    return ZipDataset({"ls_slice": get_dataset_from_info(info=info, cache=True)})


@pytest.fixture
def beads_dataset(meta) -> N5CachedDatasetFromInfo:
    datasets = OrderedDict()
    for name in ["ls_trf", "ls_reg"]:
        info = get_tensor_info("heart_static.beads_ref_wholeFOV", name, meta=meta)
        datasets[name] = get_dataset_from_info(info=info, cache=True)

    return ZipDataset(datasets)


@pytest.fixture
def scaled_metrics_config() -> typing.Dict[str, typing.Dict[str, typing.Any]]:
    return yaml.load(
        """
    MS_SSIM: {scale: pred, to_minimize: mse, vs: ls_trf, tensor_names: [pred, ls_trf], data_range: 1, size_average: true, win_size: 11, win_sigma: 1.5, channel: 1, spatial_dims: 2}
    SSIM: {scale: pred, to_minimize: mse, vs: ls_trf, tensor_names: [pred, ls_trf], data_range: 1, size_average: true, win_size: 11, win_sigma: 1.5, channel: 1, spatial_dims: 2}
    NRMSE: {scale: pred, to_minimize: mse, vs: ls_trf, tensor_names: {pred: pred, tgt: ls_trf}}
    PSNR: {scale: pred, to_minimize: mse, vs: ls_trf, tensor_names: {pred: pred, tgt: ls_trf}, data_range: 1}
    SmoothL1Loss: {scale: pred, to_minimize: mse, vs: ls_trf, tensor_names: [pred, ls_trf]}
    MSELoss: {scale: pred, to_minimize: mse, vs: ls_trf, tensor_names: [pred, ls_trf]}
    """
    )
