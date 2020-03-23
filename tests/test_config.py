import yaml

from lnet.config import Config


def test_config(dummy_config_path):
    with dummy_config_path.open() as f:
        raw_config = yaml.safe_load(f)

    config = Config(**raw_config)
    assert config
