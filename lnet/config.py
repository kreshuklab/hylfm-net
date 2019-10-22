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


def resolve_python_name_conflicts(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    trailing_underscore = ["min", "max"]
    return {k + "_" if k in trailing_underscore else k: v for k, v in kwargs.items()}


@dataclass
class LogConfig:
    config_path: InitVar[Path]

    log_scalars_every: Tuple[int, str]
    log_images_every: Tuple[int, str]
    log_bead_precision_recall: bool = False
    log_bead_precision_recall_threshold: float = 5.0
    save_n_checkpoints: int = 1

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
        cls, config_path: str, log_scalars_every: Tuple[int, str], log_images_every: Tuple[int, str], **kwargs
    ) -> "LogConfig":
        return cls(
            config_path=Path(config_path),
            log_scalars_every=tuple(log_scalars_every),
            log_images_every=tuple(log_images_every),
            **kwargs,
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

    loss: Callable[[Engine, Dict[str, Any]], List[Tuple[float, Callable]]]
    loss_kwargs: Dict[str, Any] = field(default_factory=dict)
    aux_loss: Callable[[Engine, Dict[str, Any]], Optional[List[Tuple[float, Callable]]]] = lambda *_, **__: None
    aux_loss_kwargs: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(
        cls,
        optimizer: Dict[str, Union[str, dict]],
        score_function: str,
        patience: int,
        validate_every_nth_epoch: int,
        loss: Optional[Dict[str, Union[str, Dict[str, Any]]]] = None,
        aux_loss: Optional[Dict[str, Union[str, Dict[str, Any]]]] = None,
        **kwargs,
    ) -> "TrainConfig":
        from lnet.losses import known_losses
        from lnet.optimizers import known_optimizers
        from lnet.score_functions import known_score_functions

        loss_dicts = {l: d for l, d in {"loss": loss, "aux_loss": aux_loss}.items() if d is not None}
        for loss_variant, loss_dict in loss_dicts.items():
            for key in ["name", "kwargs"]:
                if f"{loss_variant}_{key}" in kwargs:
                    raise ValueError(
                        f"invalid training key: '{loss_variant}_{key}', specify {loss_variant} as dict with {key} as key instead."
                    )

            loss_variant_name = loss_dict.pop("name")
            kwargs[loss_variant] = known_losses[loss_variant_name]
            loss_variant_kwargs = loss_dict.pop("kwargs", {})
            kwargs[f"{loss_variant}_kwargs"] = loss_variant_kwargs
            if loss_dict:
                raise ValueError(f"unknown config keys in {loss_variant}: {loss_dict}")

        return cls(
            optimizer=partial(get_known(known_optimizers, optimizer["name"]), **optimizer["kwargs"]),
            score_function=known_score_functions[score_function],
            patience=patience,
            validate_every_nth_epoch=validate_every_nth_epoch,
            **kwargs,
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
    transform_configs: List[Union[str, Dict[str, Union[str, Dict[str, Any]]]]] = None

    transform_names: List[str] = field(init=False)
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

        def get_trfs_and_their_names(
            transforms: Optional[List[Any]], conf: Optional[List[Union[str, Dict[str, Any]]]]
        ) -> Tuple[List[Callable], List[str]]:
            if transforms is None:
                if conf is None:
                    new_transforms = []
                    new_names = []
                else:
                    new_transforms = []
                    new_names = []
                    for c in conf:
                        if isinstance(c, str):
                            name = c
                            kwargs = {}
                        else:
                            assert isinstance(c, dict), type(c)
                            name = c["name"]
                            kwargs = c.get("kwargs", {})
                            left = {k: v for k, v in c.items() if k not in ["name", "kwargs"]}
                            if left:
                                raise ValueError(
                                    f"invalid keys in transformation entry with name: {name} and kwargs: {kwargs}: {left}"
                                )

                        kwargs = resolve_python_name_conflicts(kwargs)
                        new_transforms.append(get_known(known_transforms, name)(config=config, kwargs=kwargs))
                        new_names.append(name)

            elif conf is None:
                new_transforms = transforms
                new_names = [t.__name__ for t in transforms]
            else:
                new_transforms = transforms
                new_names = [c if isinstance(c, str) else c["name"] for c in conf]

            return new_transforms, new_names

        self.transforms, self.transform_names = get_trfs_and_their_names(self.transforms, self.transform_configs)
        self.transforms.append(known_transforms["Cast"](config=config, kwargs={}))

    @classmethod
    def load(
        cls: Type[DataConfigType],
        transforms: List[Union[str, Dict[str, Union[str, Dict[str, Any]]]]],
        indices: Optional[List[Optional[Union[str, int]]]] = None,
        **kwargs,
    ) -> DataConfigType:
        # note: 'transform_configs' is called 'transforms' in yaml!
        if indices is None:
            indices = [None] * len(kwargs["names"])

        return cls(transform_configs=transforms, indices=list(map(cls.indice_string_to_list, indices)), **kwargs)


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
            try:
                setattr(self, attr, DataConfig.load(config=self, name=attr, **values))
            except TypeError:
                logger.error(f"could not load {attr}")
                raise

        return self


if __name__ == "__main__":
    from_file = Config.from_yaml("experiment_configs/fish0.yml")
    # from_file = Config.from_yaml("experiment_configs/platy0.yml")
    # from_file = Config.from_yaml("experiment_configs/platy/test/1/19-09-03_09-14_63d6439_m12dout-bc.yml")
