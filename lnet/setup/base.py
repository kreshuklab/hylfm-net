from __future__ import annotations

import logging
import shutil
import subprocess
import typing
from dataclasses import InitVar, dataclass, field
from datetime import datetime
from enum import Enum
from importlib import import_module
from inspect import signature
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type, Union

import ignite
import numpy
import pbs3
import torch.utils.data
import yaml

import lnet.criteria
import lnet.metrics
import lnet.optimizers
import lnet.registration
import lnet.transforms
from lnet import settings
from lnet.criteria import CriterionWrapper
from lnet.datasets import ConcatDataset, N5ChunkAsSampleDataset, NamedDatasetInfo, collate_fn
from lnet.metrics import get_output_transform
from lnet.models import LnetModel
from lnet.step import inference_step, training_step
from lnet.transforms.base import ComposedTransform, Transform
from lnet.utils.batch_sampler import NoCrossBatchSampler
from ._utils import enforce_types

logger = logging.getLogger(__name__)


class PeriodUnit(Enum):
    epoch = "epoch"
    iteration = "iteration"


# @enforce_types
@dataclass
class Period:
    value: int
    unit: str

    def __post_init__(self):
        self.unit: PeriodUnit = PeriodUnit(self.unit)


# @enforce_types
@dataclass
class ModelSetup:
    name: str
    kwargs: Dict[str, Any]
    checkpoint: Optional[str] = None
    partial_weights: bool = False

    def __post_init__(self):
        self.checkpoint: Optional[Path] = None if self.checkpoint is None else Path(self.checkpoint)
        if self.checkpoint is not None:
            assert self.checkpoint.exists()

    def get_model(self, device: torch.device, dtype: torch.dtype) -> LnetModel:
        model_module = import_module("." + self.name.lower(), "lnet.models")
        model_class = getattr(model_module, self.name)
        model = model_class(**self.kwargs)
        model = model.to(device=device, dtype=dtype)
        if self.checkpoint is not None:
            state = torch.load(self.checkpoint, map_location=device)
            for attempt in range(2):
                try:
                    model.load_state_dict(state, strict=False)
                except RuntimeError as e:
                    if not self.partial_weights or attempt > 0:
                        raise

                    logger.warning("!!!State from checkpoint does not match!!!")

                    # load partial state
                    before = "size mismatch for "
                    after = ": copying a param with shape"
                    for line in str(e).split("\n")[1:]:
                        idx_before = line.find(before)
                        idx_after = line.find(after)
                        if idx_before == -1 or idx_after == -1:
                            logger.warning("Didn't understand 'load_state_dict' exception line: %s", line)
                        else:
                            state.pop(line[idx_before + len(before) : idx_after])
        return model


# @enforce_types
class LogSetup:
    def __init__(
        self,
        log_scalars_period: Dict[str, Union[int, str]],
        log_images_period: Dict[str, Union[int, str]],
        log_bead_precision_recall: bool = False,
        log_bead_precision_recall_threshold: float = 5.0,
        save_n_checkpoints: int = 1,
    ):
        self.log_scalars_period: Period = Period(**log_scalars_period)
        self.log_images_period: Period = Period(**log_images_period)
        self.log_bead_precision_recall = log_bead_precision_recall
        self.log_bead_precision_recall_threshold = log_bead_precision_recall_threshold
        self.save_n_checkpoints = save_n_checkpoints


# @enforce_types
class DatasetGroupSetup:
    def __init__(
        self,
        batch_size: int,
        interpolation_order: int,
        datasets: Dict[str, Dict[str, Any]],
        transforms: List[Dict[str, Any]],
        setup: Setup,
        ls_affine_transform_class: Optional[str] = None,
    ):
        self.batch_size = batch_size
        self.interpolation_order = interpolation_order
        self.setup = setup

        self.ls_affine_transform_class = (
            None if ls_affine_transform_class is None else getattr(lnet.registration, ls_affine_transform_class)
        )
        self.transforms: List[Transform] = [
            getattr(lnet.transforms, trf["name"])(**trf["kwargs"]) for trf in transforms
        ]
        batch_transform_start = numpy.argmax([trf.randomly_changes_shape for trf in self.transforms])
        self.sample_transform = ComposedTransform(*self.transforms[:batch_transform_start])
        self.batch_transform = ComposedTransform(*self.transforms[batch_transform_start:])
        self._dataset: Optional[ConcatDataset] = None
        self.datasets: Dict[str, DatasetSetup] = {
            name: DatasetSetup(**kwargs, _group_config=self, name=name) for name, kwargs in datasets.items()
        }

    def get_individual_dataset(self, name: str) -> N5ChunkAsSampleDataset:
        config = self.datasets[name]
        return N5ChunkAsSampleDataset(
            info=config.info,
            nnum=self.setup.nnum,
            z_out=self.setup.z_out,
            interpolation_order=self.interpolation_order,
            data_cache_path=self.setup.data_cache_path,
            get_model_scaling=self.setup.model.get_scaling,
            transform=self.sample_transform,
            ls_affine_transform_class=self.ls_affine_transform_class,
        )

    @property
    def dataset(self) -> ConcatDataset:
        if self._dataset is None:
            self._dataset = ConcatDataset(
                [self.get_individual_dataset(name) for name in self.datasets.keys()], transform=self.batch_transform
            )
        return self._dataset


# @enforce_types
@dataclass
class DatasetSetup:
    name: str
    indices: Union[str, int]
    _group_config: InitVar[DatasetGroupSetup]
    interpolation_order: Optional[int] = None
    transform: ComposedTransform = field(init=False)
    info: NamedDatasetInfo = field(init=False)

    def __post_init__(self, _group_config: DatasetGroupSetup):
        self.interpolation_order = None if self.interpolation_order is None else _group_config.interpolation_order
        self.transform = _group_config.sample_transform
        info_module_name, info_name = self.name.split(".")
        info_module = import_module("." + info_module_name, "lnet.datasets")
        self.info = getattr(info_module, info_name)


class DataSetup:
    def __init__(self, dataset_groups: List[DatasetGroupSetup]):
        self.groups = dataset_groups
        self._dataset = None
        self._data_loader = None

    @property
    def dataset(self) -> ConcatDataset:
        if self._dataset is None:
            self._dataset = ConcatDataset([group.dataset for group in self.groups])

        return self._dataset

    @property
    def batch_sizes(self) -> List[int]:
        return [group.batch_size for group in self.groups]


# @enforce_types
@dataclass
class SamplerSetup:
    base: str
    drop_last: bool

    _data_setup: DataSetup
    _batch_sampler: Optional[NoCrossBatchSampler] = field(init=False, default=None)

    @property
    def batch_sampler(self) -> NoCrossBatchSampler:
        if self._batch_sampler is None:
            base_sampler = getattr(torch.utils.data, self.base)
            self._batch_sampler = NoCrossBatchSampler(
                concat_dataset=self._data_setup.dataset,
                sampler_class=base_sampler,
                batch_sizes=self._data_setup.batch_sizes,
                drop_last=self.drop_last,
            )

        return self._batch_sampler


# @enforce_types
class Stage:
    step_function: Callable[[ignite.engine.Engine, Any], typing.OrderedDict[str, Any]]
    max_epochs: int = 1
    epoch_length: Optional[int] = None
    seed: Optional[int] = None

    def __init__(
        self,
        *,
        name: str,
        data: List[Dict[str, Any]],
        sampler: Dict[str, Any],
        metrics: Dict[str, Any] = None,
        setup: Setup,
    ):
        self.name = name
        self.metrics = metrics or {}
        self.data: DataSetup = DataSetup([DatasetGroupSetup(**d, setup=setup) for d in data])
        self.sampler: SamplerSetup = SamplerSetup(**sampler, _data_setup=self.data)
        self._engine: Optional[ignite.engine.Engine] = None
        self.setup = setup

    @property
    def data_loader(self) -> torch.utils.data.DataLoader:
        if self._data_loader is None:
            self._data_loader = torch.utils.data.DataLoader(
                dataset=self.data.dataset,
                batch_sampler=self.sampler.batch_sampler,
                collate_fn=collate_fn,
                num_workers=settings.num_workers_train_data_loader,
                pin_memory=settings.pin_memory,
            )

        return self._data_loader

    def prepare_engine(self, engine: ignite.engine.Engine):
        engine.state.compute_time = 0.0
        engine.state.setup = self.setup

    def log_compute_time(self, engine: ignite.engine.Engine):
        mins, secs = divmod(engine.state.compute_time / max(1, engine.state.iteration), 60)
        msecs = (secs % 1) * 1000
        hours, mins = divmod(mins, 60)
        engine.logger.info(
            "%s run on %d mini-batches completed in %.2f s with avg compute time %02d:%02d:%02d:%03d",
            self.name,
            len(engine.state.dataloader),
            engine.state.compute_time,
            hours,
            mins,
            secs,
            msecs,
        )

    def attach_metrics(self, engine: ignite.engine.Engine):
        for metric_name, kwargs in self.metrics.items():
            metric_class = getattr(lnet.metrics, metric_name, None)
            if metric_class is None:
                if (
                    isinstance(self, TrainStage)
                    and metric_name == self.criterion_setup.name
                    and kwargs == self.criterion_setup.kwargs
                ):
                    metric = ignite.metrics.Average(output_transform=lambda out: out[metric_name])
                else:
                    criterion_class = getattr(lnet.criteria, metric_name, None)
                    if criterion_class is None:
                        raise ValueError(f"{metric_name} is not a valid metric name")

                    postfix = kwargs.get("postfix", "-for-metric")
                    criterion = lnet.criteria.CriterionWrapper(criterion_class=criterion_class, **kwargs)
                    criterion.eval()
                    metric = ignite.metrics.Average(lambda tensors: tensors[criterion_class.__name__ + postfix])
            else:
                metric = metric_class(get_output_transform(kwargs.pop("tensor_names")), **kwargs)

            metric.attach(engine, metric.__name__)

    def attach_handlers(self, engine: ignite.engine.Engine):
        pass

    @property
    def log_path(self) -> Path:
        return self.setup.log_path / self.name

    @property
    def engine(self):
        if self._engine is None:
            self.log_path.mkdir(parents=True, exist_ok=False)
            engine = ignite.engine.Engine(self.step_function)
            engine.add_event_handler(ignite.engine.Events.STARTED, self.prepare_engine)
            engine.add_event_handler(ignite.engine.Events.COMPLETED, self.log_compute_time)
            self.attach_metrics(engine)
            self.attach_handlers(engine)

            self._engine = engine
        return self._engine

    def run(self):
        self.engine.run(
            data=self.data_loader, max_epochs=self.max_epochs, epoch_length=self.epoch_length, seed=self.seed
        )


# @enforce_types
class EvalStage(Stage):
    step_function = inference_step

    def __init__(self, sampler: Dict[str, Any] = None, **super_kwargs):
        if sampler is None:
            sampler = {"base": "SequentialSampler", "drop_last": False}

        super().__init__(sampler=sampler, **super_kwargs)


# @enforce_types
class ValidateStage(EvalStage):
    def __init__(
        self,
        *,
        period: Dict[str, Union[int, str]],
        patience: int,
        train_stage: TrainStage,
        metrics: Dict[str, Any],
        score_metric: str,
        **super_kwargs,
    ):
        super().__init__(metrics=metrics, **super_kwargs)
        self.period = Period(**period)
        self.patience = patience
        self.train_stage = train_stage
        assert score_metric in metrics or score_metric[1:] == train_stage.criterion_setup.name
        self.negate_score_metric = score_metric.startswith("-")
        self.score_metric = score_metric[int(self.negate_score_metric) :]

    def attach_handlers(self, engine: ignite.engine.Engine):
        super().attach_handlers(engine)
        loss_name = self.train_stage.criterion_setup.name + "-running"
        if self.score_metric[1:] == loss_name:

            def score_function(e: ignite.engine.Engine):
                return -e.state.output[loss_name]

        else:

            def score_function(e: ignite.engine.Engine):
                score = getattr(e.state, self.score_metric)
                if self.negate_score_metric:
                    score *= -1

                return score

        early_stopping = ignite.handlers.EarlyStopping(
            patience=self.patience, score_function=score_function, trainer=self.train_stage.engine
        )
        engine.add_event_handler(ignite.engine.Events.COMPLETED, early_stopping)


# @enforce_typesb
@dataclass
class CriterionSetup:
    name: str
    kwargs: Dict[str, Any]
    _class: Type[torch.nn.Module] = field(init=False)
    tensor_names: Dict[str, str]

    def __post_init__(self):
        self._class = getattr(lnet.criteria, self.name)
        assert "engine" not in self.kwargs
        assert "tensor_names" not in self.kwargs

    def get_criterion(self, *, engine: ignite.engine.Engine):
        sig = signature(self._class)
        kwargs = dict(self.kwargs)
        if "engine" in sig.parameters:
            kwargs["engine"] = engine

        return CriterionWrapper(tensor_names=self.tensor_names, criterion_class=self._class, **kwargs)


# @enforce_types
@dataclass
class OptimizerSetup:
    name: str
    kwargs: Dict[str, Any]

    def __post_init__(self):
        self._class = getattr(lnet.optimizers, self.name)
        assert "engine" not in self.kwargs
        assert "parameters" not in self.kwargs

    def get_optimizer(self, *, engine: ignite.engine.Engine):
        sig = signature(self._class)
        kwargs = dict(self.kwargs)
        if "engine" in sig.parameters:
            kwargs["engine"] = engine

        return self._class(engine.state.model.parameters(), **kwargs)


# @enforce_types
class TrainStage(Stage):
    step_function = training_step

    def __init__(
        self,
        max_num_epochs: int,
        log: Dict[str, Dict[str, Any]],
        validation_stages: Dict[str, Dict[str, Any]],
        criterion: Dict[str, Dict[str, Any]],
        optimizer: Dict[str, Dict[str, Any]],
        setup: Setup,
        **super_kwargs,
    ):
        super().__init__(setup=setup, **super_kwargs)
        self.max_num_epochs = max_num_epochs
        self.log = LogSetup(**log)
        self.criterion_setup = CriterionSetup(**criterion)
        # self._criterion: Optional[torch.nn.Module] = None
        self.optimizer_setup = OptimizerSetup(**optimizer)
        self.validation_stages = {
            ValidateStage(**config, name=name, setup=setup, train_stage=self)
            for name, config in validation_stages.items()
        }

    def prepare_engine(self, engine: ignite.engine.Engine):
        super().prepare_engine(engine)
        engine.state.optimizer = self.optimizer_setup.get_optimizer(engine=engine)
        engine.state.criterion = self.criterion_setup.get_criterion(engine=engine)
        engine.state.criterion_name = self.criterion_setup.name

    def attach_metrics(self, engine: ignite.engine.Engine):
        running_loss = ignite.metrics.RunningAverage(output_transform=lambda out: out[self.criterion_setup.name])
        running_loss.attach(engine, f"{self.criterion_setup.name}-running")

    # @property
    # def criterion(self):
    #     if self._criterion is None:
    #         self._criterion = self.criterion_setup.get_criterion()
    #     return self._criterion
    #
    # @property
    # def optimizer(self):
    #     if self._optimizer is None:
    #         self._optimizer = self.optimizer_setup.get_optimizer(self.model)


# @enforce_types
class Setup:
    def __init__(
        self,
        *,
        data_cache_path: str,
        config_path: Path,
        precision: str,
        device: Union[int, str] = 0,
        nnum: int,
        z_out: int,
        model: Dict[str, Any],
        stages: List[Dict[str, Any]],
        log_path: Optional[Path] = None,
    ):
        self.dtype: torch.dtype = getattr(torch, precision)
        assert isinstance(self.dtype, torch.dtype)
        self.data_cache_path = Path(data_cache_path)
        self.nnum = nnum
        self.z_out = z_out
        self.config_path = config_path
        self._log_path = log_path
        self._model: Optional[LnetModel] = None
        if isinstance(device, int) or "cuda" in device:
            cuda_device_count = torch.cuda.device_count()
            if cuda_device_count == 0:
                raise RuntimeError("no CUDA devices available!")
            elif cuda_device_count > 1:
                raise RuntimeError("too many CUDA devices available! (limit to one)")

        self.device = torch.device(device)
        self.model_config = ModelSetup(**model)
        self.stages = [
            {
                (
                    TrainStage(**kwargs, name=name, setup=self)
                    if "optimizer" in kwargs
                    else EvalStage(**kwargs, name=name, setup=self)
                )
                for name, kwargs in stage.items()
            }
            for stage in stages
        ]
        # self.test: EvalStage = EvalStage(
        #     **test,
        # )
        # self.train: Optional[TrainStage] = None if train is None else TrainStage(
        #     **train,
        #     _nnum=self.nnum,
        #     _z_out=self.z_out,
        #     _data_cache_path=self.data_cache_path,
        #     model=self.get_scaling,
        # )

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "Setup":
        with yaml_path.open() as f:
            config = yaml.safe_load(f)

        return cls(**config, config_path=yaml_path)

    @property
    def log_path(self) -> Path:
        if self._log_path is None:
            try:
                commit_hash = pbs3.git("rev-parse", "--verify", "HEAD").stdout
            except pbs3.CommandNotFound:
                commit_hash = subprocess.run(
                    ["git", "rev-parse", "--verify", "HEAD"], capture_output=True, text=True
                ).stdout

            log_sub_dir: List[str] = self.config_path.with_suffix("").resolve().as_posix().split("/experiment_configs/")
            assert len(log_sub_dir) == 2, log_sub_dir
            log_sub_dir: str = log_sub_dir[1]
            log_path = Path(__file__).parent / "../../logs" / log_sub_dir / datetime.now().strftime("%y-%m-%d_%H-%M-%S")
            logger.info("log_path: %s", log_path)
            log_path.mkdir(parents=True, exist_ok=True)
            (log_path / "full_commit_hash.txt").write_text(commit_hash)
            shutil.copy(self.config_path.as_posix(), (log_path / "config.yaml").as_posix())
            self._log_path = log_path

        return self._log_path

    @property
    def model(self) -> LnetModel:
        if self._model is None:
            self._model = self.model_config.get_model(device=self.device, dtype=self.dtype)

        return self._model

    def get_scaling(self, ipt_shape: Optional[Tuple[int, int]] = None) -> Tuple[float, float]:
        return self.model.get_scaling(ipt_shape)

    def run(self):
        for parallel_stages in self.stages:
            for stage in parallel_stages:
                logger.info("starting stage: %s", stage.name)
                stage.run()
