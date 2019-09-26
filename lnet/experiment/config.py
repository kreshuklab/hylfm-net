import logging
import shutil

import pbs3
import yaml

from dataclasses import dataclass, field, InitVar
from datetime import datetime
from ignite.engine import Events
from pathlib import Path
from typing import Tuple, Optional, Type, Dict, Any

from lnet import models

logger = logging.getLogger(__name__)


@dataclass
class LogConfig:
    config_path: InitVar[Path]

    commit_hash: str = field(init=False)
    time_stamp: str = field(init=False)
    dir: Path = field(init=False)

    validate_every_nth_epoch: int = 1
    log_scalars_every: Tuple[int, str] = (1, Events.ITERATION_COMPLETED)
    log_images_every: Tuple[int, str] = (1, Events.EPOCH_COMPLETED)
    log_bead_precision_recall: bool = False

    def __post_init__(self, config_path):
        self.commit_hash = pbs3.git("rev-parse", "--verify", "HEAD").stdout
        self.time_stamp = datetime.now().strftime("%y-%m-%d_%H-%M-%S")

        log_sub_dir: str = (config_path.parent / config_path.stem).absolute().as_posix().split("/config/")
        assert len(log_sub_dir) == 2, log_sub_dir
        log_sub_dir = log_sub_dir[1]
        self.dir = Path(__file__).parent.parent.parent / "logs" / log_sub_dir / self.time_stamp

        self.dir.mkdir(parents=True, exist_ok=False)

        logger.info("logging to %s", self.dir.as_posix())
        with (self.dir / "full_commit_hash.txt").open("w") as f:
            f.write(self.commit_hash)

        shutil.copy(config_path.as_posix(), self.dir.with_name(config_path.name).as_posix())


@dataclass
class ModelConfig:
    log_dir: InitVar[Path]

    name: str
    kwargs: Dict[str, Any] = field(default_factory=dict)
    checkpoint: Optional[Path] = None

    Model: Type = field(init=False)

    def __post_init__(self, log_dir):
        self.Model = getattr(models, self.name)

        if self.checkpoint is not None:
            self.checkpoint = Path(self.checkpoint)
            assert self.checkpoint.exists(), self.checkpoint
            assert self.checkpoint.is_file(), self.checkpoint
            hard_linked_checkpoint = log_dir / "checkpoint.pth"
            pbs3.ln(self.checkpoint.absolute().as_posix(), hard_linked_checkpoint.absolute().as_posix())
            self.checkpoint = hard_linked_checkpoint


class Config:
    log: LogConfig
    model: ModelConfig

    def __init__(self, config_path: Path):
        with config_path.open("r") as config_file:
            config = yaml.safe_load(config_file)

        self.log = LogConfig(config_path=config_path, **config.get("log", {}))
        self.model = ModelConfig(log_dir=self.log.dir, **config["model"])
