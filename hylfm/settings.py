import logging
import sys
from sys import platform

from .settings_default import Settings

logger = logging.getLogger(__name__)

debug_mode = getattr(sys, "gettrace", None) is not None and sys.gettrace()

if platform == "linux" or platform == "linux2":
    logger.info("using linux settings")
    settings = Settings(multiprocessing_start_method="spawn")

elif platform == "win32":
    logger.info("using windows settings")
    settings = Settings()
else:
    raise NotImplementedError
