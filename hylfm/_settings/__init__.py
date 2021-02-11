import shutil
from pathlib import Path

local_path = Path(__file__).parent / "local.py"
if not local_path.exists():
    template_path = Path(__file__).parent / "local.template.py"
    shutil.copy(template_path, local_path)

from .local import settings  # noqa
