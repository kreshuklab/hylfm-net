import logging
import pbs3
import shutil
import yaml

from dataclasses import dataclass, field, InitVar
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Tuple, Optional, Type, Dict, Any, Union, List, Callable, Generator, TypeVar

from ignite.engine import Engine
from inferno.io.transform import Transform

from lnet import models
from lnet.dataset_configs import beads, platy, fish, DatasetConfigEntry
from lnet.datasets import DatasetFactory
from lnet.transforms import known_transforms


logger = logging.getLogger(__name__)


def get_known(known: Dict[str, Any], name: str):
    res = known.get(name, None)
    if res is None:
        raise ValueError(f"{name} not known. Valid values are:\n{', '.join(known.keys())}")

    return res


@dataclass
class LogConfig:
    config_path: InitVar[Path]

    log_scalars_every: Tuple[int, str]
    log_images_every: Tuple[int, str]
    log_bead_precision_recall: bool = False
    log_bead_precision_recall_threshold: float = 5.0

    commit_hash: str = field(init=False)
    time_stamp: str = field(init=False)
    dir: Path = field(init=False)

    def __post_init__(self, config_path):
        assert self.log_scalars_every[1] in ["iteration_completed", "epoch_completed"]
        assert self.log_images_every[1] in ["iteration_completed", "epoch_completed"]

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
        log_scalars_every: Tuple[int, str],
        log_images_every: Tuple[int, str],
        log_bead_precision_recall: bool = False,
    ) -> "LogConfig":
        return cls(
            config_path=Path(config_path),
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
    validate_every_nth_epoch: int

    loss_fn: Callable[..., List[Tuple[float, Callable]]]
    loss_fn_kwargs: Dict[str, Any] = field(default_factory=dict)
    loss_fn_aux: Optional[Callable[..., List[Tuple[float, Callable]]]] = None
    loss_fn_aux_kwargs: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(
        cls,
        optimizer: Dict[str, Union[str, dict]],
        batch_size: int,
        max_num_epochs: int,
        score_function: str,
        patience: int,
        validate_every_nth_epoch: int,
        loss_fn: str,
        loss_fn_kwargs: Dict[str, Any] = None,
        loss_fn_aux: str = None,
        loss_fn_aux_kwargs: Dict[str, Any] = None,
    ) -> "TrainConfig":
        from lnet.losses import known_losses
        from lnet.optimizers import known_optimizers
        from lnet.score_functions import known_score_functions

        return cls(
            optimizer=partial(get_known(known_optimizers, optimizer["name"]), **optimizer["kwargs"]),
            batch_size=batch_size,
            max_num_epochs=max_num_epochs,
            score_function=known_score_functions[score_function],
            patience=patience,
            validate_every_nth_epoch=validate_every_nth_epoch,
            loss_fn=known_losses[loss_fn],
            loss_fn_kwargs={} if loss_fn_kwargs is None else loss_fn_kwargs,
            loss_fn_aux=None if loss_fn_aux is None else known_losses[loss_fn_aux],
            loss_fn_aux_kwargs={} if loss_fn_aux_kwargs is None else loss_fn_aux_kwargs,
        )


@dataclass
class EvalConfig:
    batch_size: int

    @classmethod
    def load(cls, **kwargs) -> "EvalConfig":
        return EvalConfig(**kwargs)


DataConfigType = TypeVar("DataConfigType", bound="DataConfig")


@dataclass
class DataConfig:
    config: InitVar["Config"]

    name: str  # name of this dataconfig
    names: List[str]  # names of datasets to be combined

    indices: Optional[List[Optional[List[int]]]] = None
    transforms: List[Union[Generator[Transform, None, None], Transform]] = None
    transform_names: List[str] = None

    factory: DatasetFactory = field(init=False)

    @staticmethod
    def range_or_single_index_to_list(indice_string_part: str) -> List[Optional[int]]:
        """
        :param indice_string_part: e.g. 37 or 0-100 or 0-100-10
        :return: e.g. [37] or [0, 1, 2, ..., 99] or [0, 10, 20, ..., 90]
        """
        ints_in_part = [None if p is None else int(p) for p in indice_string_part.split("-")]
        assert len(ints_in_part) < 4, ints_in_part
        return list(range(*ints_in_part)) if len(ints_in_part) > 1 else ints_in_part

    @staticmethod
    def indice_string_to_list(indice_string: Optional[Union[str, int]]) -> Optional[List[int]]:
        if not indice_string:
            return None
        elif isinstance(indice_string, int):
            return [indice_string]
        else:
            concatenated_indices: List[int] = []
            for part in indice_string.split("|"):
                concatenated_indices += DataConfig.range_or_single_index_to_list(part)

            return concatenated_indices

    def __post_init__(self, config):
        if self.indices is None:
            self.indices = [None] * len(self.names)

        def find_ds_config(ds_name) -> DatasetConfigEntry:
            ds = getattr(beads, ds_name, None) or getattr(fish, ds_name, None) or getattr(platy, ds_name, None)
            if ds is None:
                raise NotImplementedError(f"could not find dataset config `{ds_name}`")

            return ds

        self.factory = DatasetFactory(*map(find_ds_config, self.names))

        def get_trfs_and_their_names(transforms: Optional[List[Any]], names: Optional[List[str]]):
            if transforms is None:
                if names is None:
                    transforms = []
                    names = []
                else:
                    transforms = [known_transforms[nt](config) for nt in names]
            elif names is None:
                names = [t.__name__ for t in transforms]

            return transforms, names

        self.transforms, self.transform_names = get_trfs_and_their_names(self.transforms, self.transform_names)
        self.transforms.append(known_transforms["Cast"](config))

    @classmethod
    def load(cls: Type[DataConfigType], transforms: List[str], name: str, **kwargs) -> DataConfigType:
        # note: 'transform_names' is called 'transforms' in yaml!
        return cls(transform_names=transforms, name=name, **kwargs)


@dataclass
class Config:
    log: LogConfig
    model: ModelConfig
    eval: EvalConfig

    train: Optional[TrainConfig] = None

    train_data: Optional[DataConfig] = None
    train_eval_data: Optional[DataConfig] = None
    valid_data: Optional[DataConfig] = None
    test_data: Optional[DataConfig] = None

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

        data_configs = {attr: config.pop(attr) for attr in ["train_data", "valid_data", "test_data"] if attr in config}

        self = cls(
            log=LogConfig.load(config_path=config_path, **config.pop("log")),
            model=ModelConfig.load(**config.pop("model")),
            train=TrainConfig.load(**config.pop("train")) if "train" in config else None,
            eval=EvalConfig.load(**config.pop("eval")),
            **config,
        )

        train_data = data_configs.get("train_data", None)
        if train_data is not None:
            eval_indices = train_data.pop("eval_indices", None)
            if eval_indices is None:
                eval_indices = train_data.get("indices", None)

            train_eval_data = dict(train_data)
            train_eval_data["indices"] = eval_indices
            data_configs["train_eval_data"] = train_eval_data

        for attr, values in data_configs.items():
            setattr(self, attr,  DataConfig.load(config=self, name=attr, **values))

        return self


if __name__ == "__main__":
    from_file = Config.from_yaml("experiment_configs/fish0.yml")
    from_file = Config.from_yaml("experiment_configs/platy0.yml")
    from_file = Config.from_yaml("experiment_configs/platy/test/1/19-09-03_09-14_63d6439_m12dout-bc.yml")
