from lnet.setup import Setup


def test_setup(dummy_config_path):
    setup = Setup.from_yaml(dummy_config_path)
    train_states, test_states = setup.run()
    assert len(train_states) == 1
    assert len(test_states) == 1
    train_state, test_state = train_states.pop(), test_states.pop()