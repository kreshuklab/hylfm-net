from hylfm.setup import Setup


def test_setup(dummy_config_path):
    setup = Setup.from_yaml(dummy_config_path)
    assert setup
