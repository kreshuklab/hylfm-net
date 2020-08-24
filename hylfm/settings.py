import logging
import sys
from pathlib import Path
from sys import platform

from .settings_default import Settings

logger = logging.getLogger(__name__)

debug_mode = getattr(sys, "gettrace", None) is not None and sys.gettrace()

if platform == "linux" or platform == "linux2":
    logger.info("using linux settings")
    settings = Settings(
        multiprocessing_start_method="spawn",
        data_roots={
            "GHUFNAGELLFLenseLeNet_Microscope": Path("/g/hufnagel/LF/LenseLeNet_Microscope"),
            "GKRESHUK": Path("/g/kreshuk"),
            "logs": Path("/g/kreshuk/LF_computed/lnet/logs"),
        },
    )
elif platform == "win32":
    logger.info("using windows settings")
    settings = Settings(
        data_roots={
            "GHUFNAGELLFLenseLeNet_Microscope": Path("H:/"),
            "GKRESHUK": Path("K:/"),
            "logs": Path("K:/LF_computed/lnet/logs"),
        }
    )
else:
    raise NotImplementedError
