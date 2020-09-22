import logging
from dataclasses import dataclass
from pathlib import Path
from sys import platform

from hylfm._settings.default import Settings

logger = logging.getLogger(__name__)

if platform == "linux" or platform == "linux2":
    logger.info("using linux settings")

    settings = Settings(
        download_dir=Path("/scratch/beuttenm/hylfm/download"),
        cache_dir=Path("/scratch/beuttenm/hylfm/cache"),
        multiprocessing_start_method="spawn",
        data_roots={
            "GHUFNAGELLFLenseLeNet_Microscope": Path("/g/hufnagel/LF/LenseLeNet_Microscope"),
            "GKRESHUK": Path("/g/kreshuk"),
            "logs": Path("/g/kreshuk/LF_computed/lnet/logs"),
        },
    )


elif platform == "win32":
    logger.info("using windows settings")

    @dataclass
    class Settings(Settings):
        data_roots = {
            "GHUFNAGELLFLenseLeNet_Microscope": Path("H:/"),
            "GKRESHUK": Path("K:/"),
            "logs": Path("K:/LF_computed/lnet/logs"),
        }


else:
    raise NotImplementedError(platform)
