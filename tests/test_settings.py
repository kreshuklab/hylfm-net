import importlib.util
from pathlib import Path


def test_template():
    module_name = "local_template"
    file_path = Path(__file__).parent / "../hylfm/_settings/local.template.py"
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.settings
