import logging
import pbs3
import shutil
import yaml

from dataclasses import dataclass, field, InitVar
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Tuple, Optional, Type, Dict, Any, Union, List, Callable, Generator

from ignite.engine import Engine
from inferno.io.transform import Transform

from lnet import models
from lnet.dataset_configs import beads, platy, fish, DatasetConfigEntry
from lnet.losses import known_losses
from lnet.optimizers import known_optimizers
from lnet.score_functions import known_score_functions
from lnet.transforms import known_transforms
from lnet.utils.datasets import DatasetFactory

logger = logging.getLogger(__name__)


def get_known(known: Dict[str, Any], name: str):
    res = known.get(name, None)
    if res is None:
        raise ValueError(f"{name} not known. Valid values are:\n{', '.join(known.keys())}")

    return res


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

        log_sub_dir: List[str] = config_path.with_suffix("").absolute().as_posix().split("/experiment_configs/")
        assert len(log_sub_dir) == 2, log_sub_dir
        log_sub_dir: str = log_sub_dir[1]
        self.dir = Path(__file__).parent.parent / "logs" / log_sub_dir / self.time_stamp

        self.dir.mkdir(parents=True, exist_ok=False)
        logger.info("logging to %s", self.dir.as_posix())
        with (self.dir / "full_commit_hash.txt").open("w") as f:
            f.write(self.commit_hash)

        shutil.copy(config_path.as_posix(), (self.dir / config_path.name).as_posix())

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
    kwargs: Dict[str, Any]
    nnum: int
    precision: str
    checkpoint: Optional[Path]

    name: str = None

    def __post_init__(self):
        assert self.precision == "float" or self.precision == "half"
        if self.name is None:
            self.name = self.Model.__name__

        assert self.checkpoint is None or self.checkpoint.exists(), self.checkpoint.absolute()

    @classmethod
    def load(
        cls,
        name: str,
        nnum: int,
        kwargs: Dict[str, Any] = None,
        precision: str = "float",
        checkpoint: Optional[str] = None,
    ) -> "ModelConfig":
        if kwargs is None:
            kwargs = {}

        return cls(
            Model=getattr(models, name),
            nnum=nnum,
            kwargs=kwargs,
            name=name,
            precision=precision,
            checkpoint=None if checkpoint is None else Path(checkpoint),
        )


@dataclass
class TrainConfig:
    optimizer: Callable
    batch_size: int
    max_num_epochs: int
    score_function: Callable[[Engine], float]
    patience: int

    loss_fn: List[Tuple[float, Callable]]
    loss_aux_fn: Optional[List[Tuple[float, Callable]]] = None  # todo: make callable only, allow for kwargs?

    @classmethod
    def load(
        cls,
        optimizer: Dict[str, Union[str, dict]],
        batch_size: int,
        max_num_epochs: int,
        score_function: str,
        patience: int,
        loss_fn: str,
        loss_aux_fn: str = None,
    ) -> "TrainConfig":
        return cls(
            optimizer=partial(get_known(known_optimizers, optimizer["name"]), **optimizer["kwargs"]),
            batch_size=batch_size,
            max_num_epochs=max_num_epochs,
            score_function=known_score_functions[score_function],
            patience=patience,
            loss_fn=known_losses[loss_fn],
            loss_aux_fn=None if loss_aux_fn is None else known_losses[loss_aux_fn],
        )


@dataclass
class Config:
    log: LogConfig
    model: ModelConfig
    train: TrainConfig

    train_dataset_names: List[str]
    train_dataset_indices: List[Optional[List[int]]]
    train_eval_dataset_indices: List[Optional[List[int]]]
    train_dataset_factory: DatasetFactory = field(init=False)
    valid_dataset_names: List[str]
    valid_dataset_indices: List[Optional[List[int]]]
    valid_dataset_factory: DatasetFactory = field(init=False)
    test_dataset_names: List[str]  # todo: get test outta here!
    test_dataset_indices: List[Optional[List[int]]]
    test_dataset_factory: DatasetFactory = field(init=False)

    train_transforms: Optional[List[Union[Generator[Transform, None, None], Transform]]] = None
    train_transform_names: Optional[List[str]] = None

    eval_transforms: Optional[List[Union[Generator[Transform, None, None], Transform]]] = None
    eval_transform_names: Optional[List[str]] = None

    def __post_init__(self):
        if self.model.checkpoint is not None:
            assert self.model.checkpoint.exists(), self.model.checkpoint
            assert self.model.checkpoint.is_file(), self.model.checkpoint
            hard_linked_checkpoint = self.log.dir / "checkpoint.pth"
            pbs3.ln(self.model.checkpoint.absolute().as_posix(), hard_linked_checkpoint.absolute().as_posix())
            self.model.checkpoint = hard_linked_checkpoint

        def find_ds_config(ds_name) -> DatasetConfigEntry:
            ds = getattr(beads, ds_name, None) or getattr(fish, ds_name, None) or getattr(platy, ds_name, None)
            if ds is None:
                raise NotImplementedError(f"could not find dataset config `{ds_name}`")

            return ds

        self.train_dataset_factory = DatasetFactory(
            *map(find_ds_config, self.train_dataset_names), has_aux=self.train.loss_aux_fn is not None
        )
        self.valid_dataset_factory = DatasetFactory(
            *map(find_ds_config, self.valid_dataset_names), has_aux=self.train.loss_aux_fn is not None
        )
        self.test_dataset_factory = DatasetFactory(
            *map(find_ds_config, self.test_dataset_names), has_aux=self.train.loss_aux_fn is not None
        )

        def get_trfs_and_their_names(transforms: Optional[List[Any]], names: Optional[List[str]]):
            if transforms is None:
                if names is None:
                    transforms = []
                    names = []
                else:
                    transforms = [known_transforms[nt](self) for nt in names]
            elif names is None:
                names = [t.__name__ for t in transforms]

            return transforms, names

        self.train_transforms, self.train_transform_names = get_trfs_and_their_names(
            self.train_transforms, self.train_transform_names
        )
        self.eval_transforms, self.eval_transform_names = get_trfs_and_their_names(
            self.eval_transforms, self.eval_transform_names
        )
        self.train_transforms.append(known_transforms["Cast"](self))
        self.eval_transforms.append(known_transforms["Cast"](self))

    @classmethod
    def from_yaml(cls, config_path: Union[str, Path]) -> "Config":
        if isinstance(config_path, str):
            config_path = Path(config_path)

        with config_path.open("r") as config_file:
            config = yaml.safe_load(config_file)

        for dataset_indices in [
            "train_dataset_indices",
            "train_eval_dataset_indices",
            "valid_dataset_indices",
            "test_dataset_indices",
        ]:
            dis: Optional[List[Optional[str]]] = config.pop(dataset_indices, None)
            if dis is None:
                ds_names = config.get(dataset_indices.replace("_indice", ""), [])
                dis = [None] * len(ds_names)
            else:

                def range_or_single_index_to_list(indice_string_part: str) -> List[Optional[int]]:
                    """
                    :param indice_string_part: e.g. 37 or 0-100 or 0-100-10
                    :return: e.g. [37] or [0, 1, 2, ..., 99] or [0, 10, 20, ..., 90]
                    """
                    ints_in_part = [None if p is None else int(p) for p in indice_string_part.split("-")]
                    assert len(ints_in_part) < 4, ints_in_part
                    return list(range(*ints_in_part)) if len(ints_in_part) > 1 else ints_in_part

                def indice_string_to_list(indice_string: Optional[Union[str, int]]) -> Optional[List[int]]:
                    if not indice_string:
                        return None
                    elif isinstance(indice_string, int):
                        return [indice_string]
                    else:
                        concatenated_indices: List[int] = []
                        for part in indice_string.split("|"):
                            concatenated_indices += range_or_single_index_to_list(part)

                        return concatenated_indices

                dis = [indice_string_to_list(d) for d in dis]

            config[dataset_indices] = dis

        named_keys = {
            "train_dataset_names": ("train_datasets", None),
            "valid_dataset_names": ("valid_datasets", None),
            "test_dataset_names": ("test_datasets", None),
            "train_transform_names": ("train_transforms", None),
            "eval_transform_names": ("eval_transforms", None),
        }
        for config_name, (yaml_name, default) in named_keys.items():
            config[config_name] = config.pop(yaml_name, default)

        return cls(
            log=LogConfig.load(config_path=config_path, **config.pop("log")),
            model=ModelConfig.load(**config.pop("model")),
            train=TrainConfig.load(**config.pop("train")),
            **config,
        )


if __name__ == "__main__":
    from_file = Config.from_yaml("experiment_configs/fish0.yml")
