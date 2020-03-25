import yaml

from lnet.setup import Setup


def test_setup(dummy_config_path):
    with dummy_config_path.open() as f:
        config = yaml.safe_load(f)

    setup = Setup(**config)
    assert setup
