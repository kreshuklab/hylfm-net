import logging

import pbs3
import shutil
import yaml

from dataclasses import dataclass, field, InitVar
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, Union, List, Callable

from ignite.engine import Engine, Events

from lnet.config.data import DataConfig, DataCategory
from lnet.config.model import ModelConfig
from lnet.config.utils import get_known

logger = logging.getLogger(__name__)


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
        assert self.log_scalars_every[1] in [e.value for e in [Events.ITERATION_COMPLETED, Events.EPOCH_COMPLETED]]
        assert self.log_images_every[1] in [e.value for e in [Events.ITERATION_COMPLETED, Events.EPOCH_COMPLETED]]

        self.commit_hash = pbs3.git("rev-parse", "--verify", "HEAD").stdout
        self.time_stamp = datetime.now().strftime("%y-%m-%d_%H-%M-%S")

        log_sub_dir: List[str] = config_path.with_suffix("").absolute().as_posix().split("/experiment_configs/")
        assert len(log_sub_dir) == 2, log_sub_dir
        log_sub_dir: str = log_sub_dir[1]
        self.dir = Path(__file__).parent.parent.parent / "logs" / log_sub_dir / self.time_stamp

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
class TrainConfig:
    optimizer: Callable
    max_num_epochs: int
    score_function: Callable[[Engine], float]
    patience: int
    validate_every_nth_epoch: int

    loss: Callable[[Engine, Dict[str, Any]], List[Tuple[float, Callable]]]
    loss_kwargs: Dict[str, Any] = field(default_factory=dict)
    aux_loss: Callable[[Engine, Dict[str, Any]], Optional[List[Tuple[float, Callable]]]] = lambda *_, **__: None
    aux_loss_kwargs: Dict[str, Any] = field(default_factory=dict)

    data: Optional[DataConfig] = None

    @classmethod
    def load(
        cls,
        model_config: ModelConfig,
        data: Dict[str, Dict[str, Any]],
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
                        f"invalid training key: '{loss_variant}_{key}', specify {loss_variant} as dict with {key} as "
                        "key instead."
                    )

            loss_variant_name = loss_dict.pop("name")
            kwargs[loss_variant] = known_losses[loss_variant_name]
            loss_variant_kwargs = loss_dict.pop("kwargs", {})
            kwargs[f"{loss_variant}_kwargs"] = loss_variant_kwargs
            if loss_dict:
                raise ValueError(f"unknown config keys in {loss_variant}: {loss_dict}")

        return cls(
            data=DataConfig.load(
                model_config=model_config,
                category=DataCategory.train,
                default_batch_size=data.pop("batch_size", None),
                default_transforms=data.pop("transforms", None),
                entries=data,
            ),
            optimizer=partial(get_known(known_optimizers, optimizer["name"]), **optimizer["kwargs"]),
            score_function=known_score_functions[score_function],
            patience=patience,
            validate_every_nth_epoch=validate_every_nth_epoch,
            **kwargs,
        )


@dataclass
class EvalConfig:
    eval_train_data: Optional[DataConfig] = None
    valid_data: Optional[DataConfig] = None
    test_data: Optional[DataConfig] = None

    @classmethod
    def load(
        cls,
        model_config: ModelConfig,
        eval_train_data: Optional[Dict[str, Any]] = None,
        valid_data: Optional[Dict[str, Any]] = None,
        test_data: Optional[Dict[str, Any]] = None,
        # general eval default:
        batch_size: Optional[int] = None,
        transforms: Optional[List[Union[str, Dict[str, Union[str, Dict[str, Any]]]]]] = None,
    ) -> "EvalConfig":
        defaults = {"batch_size": batch_size, "transforms": transforms}
        eval_kwargs = {}
        for category, data_kwargs in [
            (DataCategory.eval_train, eval_train_data),
            (DataCategory.valid, valid_data),
            (DataCategory.test, test_data),
        ]:
            if data_kwargs is None:
                data_kwargs = {}

            for default, value in defaults.items():
                if default not in data_kwargs:
                    data_kwargs[default] = value

            eval_kwargs[category.value] = DataConfig.load(
                model_config=model_config,
                category=category,
                default_batch_size=data_kwargs.pop("batch_size", None),
                default_transforms=data_kwargs.pop("transforms", None),
                entries=data_kwargs,
            )

        return EvalConfig(**eval_kwargs)


@dataclass
class Config:
    log: LogConfig
    model: ModelConfig
    eval_: EvalConfig

    train: Optional[TrainConfig] = None

    def __post_init__(self):
        if self.model.checkpoint is not None:
            hard_linked_checkpoint = self.log.dir / "checkpoint.pth"
            pbs3.ln(self.model.checkpoint.absolute().as_posix(), hard_linked_checkpoint.absolute().as_posix())
            self.model.checkpoint = hard_linked_checkpoint

    @classmethod
    def from_yaml(cls, config_path: Union[str, Path]) -> "Config":
        if isinstance(config_path, str):
            config_path = Path(config_path)

        with config_path.open("r") as config_file:
            config = yaml.safe_load(config_file)

        # data_configs = {attr: config.pop(attr) for attr in ["train_data", "valid_data", "test_data"] if attr in config}
        #
        # train_data = data_configs.get("train_data", None)
        # if train_data is not None:
        #     eval_indices = train_data.pop("eval_indices", None)
        #     if eval_indices is None:
        #         eval_indices = train_data.get("indices", None)
        #
        #     train_eval_data = dict(train_data)
        #     train_eval_data["indices"] = eval_indices
        #     data_configs["train_eval_data"] = train_eval_data

        model_config = ModelConfig.load(**config.pop("model"))
        log_config = LogConfig.load(config_path=config_path, **config.pop("log"))
        eval_config = EvalConfig.load(model_config=model_config, **config.pop("eval"))
        train_config = TrainConfig.load(model_config=model_config, **config.pop("train")) if "train" in config else None

        return cls(log=log_config, model=model_config, train=train_config, eval_=eval_config)


if __name__ == "__main__":
    from_file = Config.from_yaml("experiment_configs/fish0.yml")
    print(from_file)
    print()
    # from_file = Config.from_yaml("experiment_configs/platy0.yml")
    # from_file = Config.from_yaml("experiment_configs/platy/test/1/19-09-03_09-14_63d6439_m12dout-bc.yml")
