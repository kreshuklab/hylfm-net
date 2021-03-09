from ._version import get_versions
from ._settings import settings

__version__ = get_versions()["version"]
del get_versions
