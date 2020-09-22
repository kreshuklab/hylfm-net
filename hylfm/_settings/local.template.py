import logging
import sys
from pathlib import Path

from hylfm._settings.default import Settings

logger = logging.getLogger(__name__)

settings = Settings(
    # download_dir=Path("/my/download/dir"),
)
