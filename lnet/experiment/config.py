import logging
import shutil

import pbs3
import yaml

from dataclasses import dataclass, field, InitVar
from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional, Type, Dict, Any, Union, List, Callable, Generator

from inferno.io.transform import Transform

import lnet.transforms

from lnet import models
from lnet.utils.loss import known_losses

logger = logging.getLogger(__name__)


@dataclass
class LogConfig:
    config_path: InitVar[Path]

    validate_every_nth_epoch: int
    log_scalars_every: Tuple[int, str]
    log_images_every: Tuple[int, str]
    log_bead_precision_recall: bool = False

    commit_hash: str = field(init=False)
    time_stamp: str = field(init=False)
    dir: Path = field(init=False)

    def __post_init__(self, config_path):
        self.commit_hash = pbs3.git("rev-parse", "--verify", "HEAD").stdout
        self.time_stamp = datetime.now().strftime("%y-%m-%d_%H-%M-%S")

        log_sub_dir: str = (config_path.parent / config_path.stem).absolute().as_posix().split("/experiment_configs/")
        assert len(log_sub_dir) == 2, log_sub_dir
        log_sub_dir = log_sub_dir[1]
        self.dir = Path(__file__).parent.parent.parent / "logs" / log_sub_dir / self.time_stamp

        self.dir.mkdir(parents=True, exist_ok=False)

        logger.info("logging to %s", self.dir.as_posix())
        with (self.dir / "full_commit_hash.txt").open("w") as f:
            f.write(self.commit_hash)

        shutil.copy(config_path.as_posix(), self.dir.with_name(config_path.name).as_posix())

    @classmethod
    def load(
        cls,
        config_path: str,
        validate_every_nth_epoch: int,
        log_scalars_every: Tuple[int, str],
        log_images_every: Tuple[int, str],
        log_bead_precision_recall: bool = False,
    ) -> "LogConfig":
        return cls(
            config_path=Path(config_path),
            validate_every_nth_epoch=validate_every_nth_epoch,
            log_scalars_every=tuple(log_scalars_every),
            log_images_every=tuple(log_images_every),
            log_bead_precision_recall=log_bead_precision_recall,
        )


@dataclass
class ModelConfig:
    Model: Type
    nnum: int

    kwargs: Dict[str, Any] = field(default_factory=dict)
    name: str = None
    checkpoint: Optional[Path] = None

    def __post_init__(self):
        if self.name is None:
            self.name = self.Model.__name__

        assert self.checkpoint.exists(), self.checkpoint.absolute()

    @classmethod
    def load(
        cls, name: str, nnum: int, kwargs: Dict[str, Any] = None, checkpoint: Optional[str] = None
    ) -> "ModelConfig":
        if kwargs is None:
            kwargs = {}

        return cls(Model=getattr(models, name), nnum=nnum, kwargs=kwargs, name=name, checkpoint=Path(checkpoint))


@dataclass
class TrainConfig:
    model_config: InitVar[ModelConfig]

    loss_fn: List[Tuple[float, Callable]]
    loss_aux_fn: Optional[List[Tuple[float, Callable]]] = None  # todo: make callable only, allow for kwargs?
    transforms: Optional[List[Generator[Transform, None, None], Transform]] = None
    named_transforms: Optional[List[str]] = None

    def __post_init__(self, model_config):
        if self.transforms is None:
            if self.named_transforms is None:
                self.transforms = []
                self.named_transforms = []
            else:
                self.transforms = [getattr(lnet.transforms, nt)(model_config) for nt in self.named_transforms]

    @classmethod
    def load(cls, model_config: ModelConfig, transforms: List[str], loss_fn: str, loss_aux_fn: str) -> "TrainConfig":
        return cls(
            model_config=model_config,
            named_transforms=transforms,
            loss_fn=known_losses[loss_fn],
            loss_aux_fn=known_losses[loss_aux_fn],
        )


@dataclass
class DataConfig:
    normalization: str

    @classmethod
    def load(cls, normalization: str) -> "DataConfig":
        return cls(normalization=normalization)


@dataclass
class Config:
    log: LogConfig
    model: ModelConfig
    train: TrainConfig
    data: DataConfig

    def __post_init__(self):
        if self.model.checkpoint is not None:
            assert self.model.checkpoint.exists(), self.model.checkpoint
            assert self.model.checkpoint.is_file(), self.model.checkpoint
            hard_linked_checkpoint = self.log.dir / "checkpoint.pth"
            pbs3.ln(self.model.checkpoint.absolute().as_posix(), hard_linked_checkpoint.absolute().as_posix())
            self.model.checkpoint = hard_linked_checkpoint

    @classmethod
    def from_yaml(cls, config_path: Union[str, Path]) -> "Config":
        if isinstance(config_path, str):
            config_path = Path(config_path)

        with config_path.open("r") as config_file:
            config = yaml.safe_load(config_file)

        return cls(
            log=LogConfig.load(config_path=config_path, **config.get("log", {})),
            model=ModelConfig.load(**config["model"]),
            train=TrainConfig.load(**config["train"]),
            data=DataConfig.load(**config["data"]),
        )


if __name__ == "__main__":
    from_file = Config.from_yaml("../../experiment_configs/fish0.yml")
