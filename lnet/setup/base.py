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
import pbs3
import torch.utils.data
import yaml

import lnet.criteria
import lnet.log
import lnet.metrics
import lnet.optimizers
import lnet.registration
import lnet.transforms
from lnet import settings
from lnet.criteria import CriterionWrapper
from lnet.datasets import ConcatDataset, N5ChunkAsSampleDataset, NamedDatasetInfo, get_collate_fn
from lnet.metrics import get_output_transform
from lnet.models import LnetModel
from lnet.step import inference_step, training_step
from lnet.transforms.base import ComposedTransform, Transform
from lnet.utils.batch_sampler import NoCrossBatchSampler

logger = logging.getLogger(__name__)


class PeriodUnit(Enum):
    epoch = "epoch"
    iteration = "iteration"


@dataclass
class Period:
    value: int
    unit: str

    def __post_init__(self):
        self.unit: PeriodUnit = PeriodUnit(self.unit)


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


class LogSetup:
    def __init__(
        self,
        *,
        log_scalars_period: Dict[str, Union[int, str]],
        log_images_period: Dict[str, Union[int, str]],
        stage: Stage,
        backend: str = "TensorBoadEvalLogger",
        **kwargs,
    ):
        self.log_scalars_period: Period = Period(**log_scalars_period)
        self.log_images_period: Period = Period(**log_images_period)
        self.stage = stage
        self.backend = getattr(lnet.log, backend)(stage=self.stage, **kwargs)

    def register_callbacks(self, engine: ignite.engine.Engine):
        pass


class TrainLogSetup(LogSetup):
    def __init__(self, save_n_checkpoints: int = 1, backend: str = "TensorBoadTrainLogger", **super_kwargs):
        super().__init__(backend=backend, **super_kwargs)
        self.save_n_checkpoints = save_n_checkpoints


class DatasetGroupSetup:
    def __init__(
        self,
        batch_size: int,
        interpolation_order: int,
        datasets: Dict[str, Dict[str, Any]],
        sample_transforms: List[Dict[str, Dict[str, Any]]],
        setup: Setup,
        ls_affine_transform_class: Optional[str] = None,
    ):
        self.batch_size = batch_size
        self.interpolation_order = interpolation_order
        self.setup = setup

        self.ls_affine_transform_class = (
            None if ls_affine_transform_class is None else getattr(lnet.registration, ls_affine_transform_class)
        )
        sample_transform_instances: List[Transform] = [
            getattr(lnet.transforms, name)(**kwargs) for trf in sample_transforms for name, kwargs in trf.items()
        ]
        assert not any([trf.randomly_changes_shape for trf in sample_transform_instances])
        self.sample_transform = ComposedTransform(*sample_transform_instances)
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
                [self.get_individual_dataset(name) for name in self.datasets.keys()], transform=None
            )
        return self._dataset


@dataclass
class DatasetSetup:
    name: str
    indices: Union[str, int]
    _group_config: InitVar[DatasetGroupSetup]
    interpolation_order: Optional[int] = None
    info: NamedDatasetInfo = field(init=False)

    def __post_init__(self, _group_config: DatasetGroupSetup):
        self.interpolation_order = None if self.interpolation_order is None else _group_config.interpolation_order
        info_module_name, info_name = self.name.split(".")
        info_module = import_module("." + info_module_name, "lnet.datasets")
        self.info = getattr(info_module, info_name)


class DataSetup:
    def __init__(self, dataset_groups: List[DatasetGroupSetup]):
        self.groups = dataset_groups
        self._dataset: Optional[ConcatDataset] = None

    @property
    def dataset(self) -> ConcatDataset:
        if self._dataset is None:
            self._dataset = ConcatDataset([group.dataset for group in self.groups])

        return self._dataset

    @property
    def batch_sizes(self) -> List[int]:
        return [group.batch_size for group in self.groups]

    def shutdown(self):
        if self._dataset is not None:
            self._dataset.shutdown()


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


class Stage:
    step_function: Callable[[ignite.engine.Engine, typing.OrderedDict[str, Any]], typing.OrderedDict[str, Any]]
    max_epochs: int = 1
    epoch_length: Optional[int] = None
    seed: Optional[int] = None

    log: LogSetup
    log_class: Type[LogSetup]

    def __init__(
        self,
        *,
        name: str,
        data: List[Dict[str, Any]],
        sampler: Dict[str, Any],
        metrics: Dict[str, Any],
        log: Dict[str, Any],
        batch_transforms: List[Dict[str, Dict[str, Any]]],
        setup: Setup,
        outputs_to_save: Optional[Sequence[str]] = tuple(),
    ):
        self.name = name
        self.outputs_to_save = outputs_to_save
        self.metrics = metrics
        self._data_loader: Optional[torch.utils.data.DataLoader] = None
        self._engine: Optional[ignite.engine.Engine] = None
        self.setup = setup
        batch_transform_instances: List[Transform] = [
            getattr(lnet.transforms, name)(**kwargs) for trf in batch_transforms for name, kwargs in trf.items()
        ]
        self.batch_transform = ComposedTransform(*batch_transform_instances)
        self.data: DataSetup = DataSetup([DatasetGroupSetup(**d, setup=setup) for d in data])
        self.sampler: SamplerSetup = SamplerSetup(**sampler, _data_setup=self.data)
        self.log = self.log_class(stage=self, **log)

    @property
    def data_loader(self) -> torch.utils.data.DataLoader:
        if self._data_loader is None:
            self._data_loader = torch.utils.data.DataLoader(
                dataset=self.data.dataset,
                batch_sampler=self.sampler.batch_sampler,
                collate_fn=get_collate_fn(
                    batch_transform=self.batch_transform, dtype=self.setup.dtype, device=self.setup.device
                ),
                num_workers=settings.num_workers_train_data_loader,
                pin_memory=settings.pin_memory,
            )

        return self._data_loader

    def start_engine(self, engine: ignite.engine.Engine):
        engine.state.compute_time = 0.0
        engine.state.stage = self

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

    def shutdown(self):
        self.data.shutdown()

    def setup_engine(self, engine: ignite.engine.Engine):
        engine.add_event_handler(ignite.engine.Events.STARTED, self.start_engine)
        engine.add_event_handler(ignite.engine.Events.COMPLETED, self.log_compute_time)

        initialized_metrics = {}
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
                        metric_getter = getattr(lnet.metrics, "get_" + metric_name, None)
                        if metric_getter is None:
                            raise ValueError(f"{metric_name} is not a valid metric name")

                        metric = metric_getter(initialized_metrics=initialized_metrics, kwargs=kwargs)
                    else:
                        postfix = kwargs.get("postfix", "-for-metric")
                        criterion = lnet.criteria.CriterionWrapper(
                            criterion_class=criterion_class, postfix=postfix, **kwargs
                        )
                        criterion.eval()
                        metric = ignite.metrics.Average(lambda tensors: tensors[criterion_class.__name__ + postfix])
            else:
                try:
                    metric = metric_class(output_transform=get_output_transform(kwargs.pop("tensor_names")), **kwargs)
                except Exception:
                    logger.error("Cannot init %s", metric_name)
                    raise

            metric.attach(engine, metric.__name__)

    @property
    def engine(self):
        if self._engine is None:
            self.log_path.mkdir(parents=True, exist_ok=False)
            self._engine = ignite.engine.Engine(self.step_function)
            self.setup_engine(self._engine)
            self.log.register_callbacks(engine=self._engine)

        return self._engine

    @property
    def log_path(self) -> Path:
        return self.setup.log_path / self.name

    def run(self):
        return self.engine.run(
            data=self.data_loader, max_epochs=self.max_epochs, epoch_length=self.epoch_length, seed=self.seed
        )


class EvalStage(Stage):
    step_function = staticmethod(inference_step)
    log: LogSetup
    log_class = LogSetup

    def __init__(self, *, sampler: Dict[str, Any] = None, **super_kwargs):
        if sampler is None:
            sampler = {"base": "SequentialSampler", "drop_last": False}

        super().__init__(sampler=sampler, **super_kwargs)


class ValidateStage(EvalStage):
    def __init__(
        self,
        *,
        period: Dict[str, Union[int, str]],
        patience: int,
        train_stage: TrainStage,
        metrics: Dict[str, Any],
        score_metric: Optional[str] = None,
        **super_kwargs,
    ):
        super().__init__(metrics=metrics, **super_kwargs)
        self.period = Period(**period)
        self.patience = patience
        self.train_stage = train_stage
        assert score_metric in metrics or score_metric[1:] == train_stage.criterion_setup.name
        self.negate_score_metric = score_metric.startswith("-")
        self.score_metric = score_metric[int(self.negate_score_metric) :]

    # def attach_handlers(self, engine: ignite.engine.Engine):
    #     super().attach_handlers(engine)
    #     loss_name = self.train_stage.criterion_setup.name + "-running"
    #     if self.score_metric[1:] == loss_name:
    #
    #         def score_function(e: ignite.engine.Engine):
    #             return -e.state.output[loss_name]
    #
    #     else:
    #
    #         def score_function(e: ignite.engine.Engine):
    #             score = getattr(e.state, self.score_metric)
    #             if self.negate_score_metric:
    #                 score *= -1
    #
    #             return score
    #
    #     early_stopping = ignite.handlers.EarlyStopping(
    #         patience=self.patience, score_function=score_function, trainer=self.train_stage.engine
    #     )
    #     engine.add_event_handler(ignite.engine.Events.COMPLETED, early_stopping)


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

        return self._class(engine.state.stage.setup.model.parameters(), **kwargs)


class TrainStage(Stage):
    log: TrainLogSetup
    log_class = TrainLogSetup
    step_function = staticmethod(training_step)

    def __init__(
        self,
        max_num_epochs: int,
        validate: Dict[str, Any],
        criterion: Dict[str, Dict[str, Any]],
        optimizer: Dict[str, Dict[str, Any]],
        setup: Setup,
        **super_kwargs,
    ):
        super().__init__(setup=setup, **super_kwargs)
        self.max_num_epochs = max_num_epochs
        self.criterion_setup = CriterionSetup(**criterion)
        self.optimizer_setup = OptimizerSetup(**optimizer)
        self.validate = ValidateStage(name="validate", setup=setup, train_stage=self, **validate)

    def setup_engine(self, engine: ignite.engine.Engine):
        super().setup_engine(engine)
        running_loss = ignite.metrics.RunningAverage(output_transform=lambda out: out[self.criterion_setup.name])
        running_loss.attach(engine, f"{self.criterion_setup.name}-running")

        if self.validate.period.unit == PeriodUnit.epoch:
            event = ignite.engine.Events.EPOCH_COMPLETED
        elif self.validate.period.unit == PeriodUnit.iteration:
            event = ignite.engine.Events.ITERATION_COMPLETED
        else:
            raise NotImplementedError

        checkpointer = ignite.handlers.ModelCheckpoint(
            (self.log_path / "checkpoints").as_posix(),
            "v1",
            score_function=lambda e: e.state.validation_score,
            score_name=self.validate.score_metric,
            n_saved=self.log.save_n_checkpoints,
            create_dir=True,
        )

        early_stopper = ignite.handlers.EarlyStopping(
            patience=self.validate.patience, score_function=lambda e: e.state.validation_score, trainer=self.engine
        )

        @engine.on(event(every=self.validate.period.value))
        def validate(e: ignite.engine.Engine):
            validation_state = self.validate.run()
            validation_score = getattr(validation_state, self.validate.score_metric)
            e.state.validation_score = validation_score
            checkpointer(
                e, {"model": e.state.model, "optimizer": e.state.optimizer, "criterion": e.state.criterion, "engine": e}
            )
            early_stopper(e)

        @engine.on(ignite.engine.Events.COMPLETED)
        def load_best_model(e: ignite.engine.Engine):
            best_score, best_checkpoint = checkpointer._saved[-1]
            assert best_score == max([item.priority for item in checkpointer._saved])
            model: torch.nn.Module = e.state.model
            model.load_state_dict(state_dict=torch.load(best_checkpoint["model"]))

    def start_engine(self, engine: ignite.engine.Engine):
        super().start_engine(engine)
        engine.state.optimizer = self.optimizer_setup.get_optimizer(engine=engine)
        engine.state.criterion = self.criterion_setup.get_criterion(engine=engine)
        engine.state.criterion_name = self.criterion_setup.name

    def shutdown(self):
        self.validate.shutdown()
        super().shutdown()


class Setup:
    def __init__(
        self,
        *,
        config_path: Path,
        precision: str,
        device: Union[int, str] = 0,
        nnum: int,
        z_out: int,
        model: Dict[str, Any],
        stages: List[Dict[str, Any]],
        data_cache_path: Optional[str] = None,
        log_path: Optional[str] = None,
    ):
        self.dtype: torch.dtype = getattr(torch, precision)
        assert isinstance(self.dtype, torch.dtype)
        self.nnum = nnum
        self.z_out = z_out
        self.config_path = config_path
        self._data_cache_path = None if data_cache_path is None else Path(data_cache_path)
        self._log_path = None if log_path is None else Path(log_path)
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

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "Setup":
        with yaml_path.open() as f:
            config = yaml.safe_load(f)

        return cls(**config, config_path=yaml_path)

    @property
    def data_cache_path(self) -> Path:
        if self._data_cache_path is None:
            self._data_cache_path = Path(__file__).parent / "../../data"
            self._data_cache_path.mkdir(exist_ok=True)

        return self._data_cache_path

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
            self._log_path = (
                Path(__file__).parent / "../../logs" / log_sub_dir / datetime.now().strftime("%y-%m-%d_%H-%M-%S")
            )
            logger.info("log_path: %s", self._log_path)
            self._log_path.mkdir(parents=True, exist_ok=False)
            (self._log_path / "full_commit_hash.txt").write_text(commit_hash)
            shutil.copy(self.config_path.as_posix(), (self._log_path / "config.yaml").as_posix())

        return self._log_path

    @property
    def model(self) -> LnetModel:
        if self._model is None:
            self._model = self.model_config.get_model(device=self.device, dtype=self.dtype)

        return self._model

    def get_scaling(self, ipt_shape: Optional[Tuple[int, int]] = None) -> Tuple[float, float]:
        return self.model.get_scaling(ipt_shape)

    def run(self) -> typing.List[typing.Set]:
        states = []
        try:
            for parallel_stages in self.stages:
                states.append(set())
                for stage in parallel_stages:
                    logger.info("starting stage: %s", stage.name)
                    states[-1].add(stage.run())
        finally:
            self.shutdown()

        return states

    def shutdown(self):
        [stage.shutdown() for parallel_stages in self.stages for stage in parallel_stages]
