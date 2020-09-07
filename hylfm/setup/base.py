from __future__ import annotations

import copy
import logging
import os
import shutil
import typing
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from functools import partial
from importlib import import_module
from inspect import signature
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type, Union

import ignite
import torch.utils.data
from ignite.handlers import global_step_from_engine
from ruamel.yaml import YAML

import hylfm.log
import hylfm.losses
import hylfm.metrics
import hylfm.optimizers
import hylfm.transformations
from hylfm import post_run, settings
from hylfm.datasets import ConcatDataset, ZipDataset, get_collate_fn, get_dataset_from_info, get_tensor_info
from hylfm.datasets.base import TensorInfo
from hylfm.losses.on_tensors import LossOnTensors
from hylfm.metrics import get_metric
from hylfm.metrics.base import Metric, MetricValue
from hylfm.models import LnetModel
from hylfm.setup._utils import indice_string_to_list
from hylfm.step import inference_step, training_step
from hylfm.transformations import ComposedTransformation, Transform
from hylfm.transformations.utils import get_composed_transformation_from_config
from hylfm.utils import Period, PeriodUnit
from hylfm.utils.batch_sampler import NoCrossBatchSampler
from hylfm.utils.general import camel_to_snake, delete_empty_dirs

logger = logging.getLogger(__name__)

yaml = YAML(typ="safe")


class ModelSetup:
    def __init__(
        self,
        name: str,
        kwargs: Dict[str, Any],
        checkpoint: Optional[Union[Path, str]] = None,
        partial_weights: bool = False,
    ):
        self.name = name
        self.kwargs = kwargs
        self.checkpoint = None if checkpoint is None else str(checkpoint)
        self.partial_weights = partial_weights
        self.strict = True

    def get_model(self, device: torch.device, dtype: torch.dtype) -> LnetModel:
        model_module = import_module("." + self.name.lower(), "hylfm.models")
        model_class = getattr(model_module, self.name)
        model = model_class(**self.kwargs)
        model = model.to(device=device, dtype=dtype)
        if self.checkpoint is not None:
            state = torch.load(self.checkpoint, map_location=device)["model"]
            if self.strict:
                model.load_state_dict(state, strict=True)
            else:
                for attempt in range(3):
                    try:
                        model.load_state_dict(state, strict=not bool(attempt))
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


class PerLoggerSetup:
    def __init__(
        self,
        name: str,
        stage: Stage,
        scalars_every: Optional[Dict[str, Union[int, str]]] = None,
        tensors_every: Optional[Dict[str, Union[int, str]]] = None,
        tensor_names: Optional[Sequence[str]] = None,
    ):
        self.backend: hylfm.log.BaseLogger = getattr(hylfm.log, name)(stage=stage, tensor_names=tensor_names)
        self.scalars_every = None if scalars_every is None else Period(**scalars_every)
        self.tensors_every = None if tensors_every is None else Period(**tensors_every)

    def register_callbacks(self, engine: ignite.engine.Engine):

        self.backend.register_callbacks(
            engine=engine, scalars_every=self.scalars_every, tensors_every=self.tensors_every
        )

    def shutdown(self):
        self.backend.shutdown()


class LogSetup:
    def __init__(self, *, stage: Stage, **loggers: dict):
        if os.environ.get("SLURM_JOB_ID", None) is not None:
            loggers.pop("TqdmLogger", None)  # no TqdmLogger in slurm jobs (progress bar spams output)

        self.loggers: Dict[str, PerLoggerSetup] = {name: PerLoggerSetup(name=name, stage=stage, **lgr) for name, lgr in loggers.items()}

    def register_callbacks(self, engine: ignite.engine.Engine):
        [lgr.register_callbacks(engine) for lgr in self.loggers.values()]

    def shutdown(self):
        [lgr.shutdown() for lgr in self.loggers.values()]


class TrainLogSetup(LogSetup):
    def __init__(self, save_n_checkpoints: int = 1, **super_kwargs):
        super().__init__(**super_kwargs)
        self.save_n_checkpoints = save_n_checkpoints


class DatasetGroupSetup:
    def __init__(
        self,
        *,
        batch_size: int,
        datasets: List[Dict[str, Any]],
        sample_preprocessing: List[Dict[str, Dict[str, Any]]],
        filters: Sequence[Tuple[str, Dict[str, Any]]] = tuple(),
        sample_transformations: Sequence[Dict[str, Dict[str, Any]]] = tuple(),
    ):
        self.batch_size = batch_size

        self.sample_preprocessing = get_composed_transformation_from_config(sample_preprocessing)
        self._dataset: Optional[ConcatDataset] = None
        self.filters = list(filters)
        for ds in datasets:

            if "sample_transformations" not in ds:
                ds["sample_transformations"] = sample_transformations

        self.dataset_setups: List[DatasetSetup] = [DatasetSetup(**kwargs) for kwargs in datasets]

    def get_individual_dataset(self, dss: DatasetSetup) -> torch.utils.data.Dataset:
        return ZipDataset(
            OrderedDict(
                [
                    (
                        name,
                        get_dataset_from_info(
                            dsinfo, cache=True, indices=dss.indices, filters=dss.filters + self.filters
                        ),
                    )
                    for name, dsinfo in dss.infos.items()
                ]
            ),
            transformation=self.sample_preprocessing,
        )

    @property
    def dataset(self) -> ConcatDataset:
        if self._dataset is None:
            self._dataset = ConcatDataset(
                [self.get_individual_dataset(dss) for dss in self.dataset_setups], transform=None
            )
        return self._dataset


class DatasetSetup:
    def __init__(
        self,
        *,
        tensors: Dict[str, Union[str, dict]],
        indices: Optional[Union[str, int, List[int]]] = None,
        filters: Sequence[Tuple[str, Dict[str, Any]]] = tuple(),
        sample_transformations: Sequence[Dict[str, Dict[str, Any]]] = tuple(),
    ):
        self.filters = list(filters)
        expected_tensor_names = set(kwargs["apply_to"] for strf in sample_transformations for kwargs in strf.values())
        assert all(
            [isinstance(etn, str) for etn in expected_tensor_names]
        ), f"sample transformations have to be applied to individual tensors, but got: {sample_transformations}"
        self.infos = OrderedDict()
        found_tensor_names = set(tensors.keys())
        if not expected_tensor_names.issubset(found_tensor_names):
            raise ValueError(
                f"Cannot apply transformations to unspecified tensors: {expected_tensor_names - found_tensor_names}"
            )
        meta = tensors.pop("meta", {})
        for name, info_name in tensors.items():
            if isinstance(info_name, str):
                info = get_tensor_info(info_name=info_name, name=name, meta=meta)
            elif isinstance(info_name, dict):
                info = TensorInfo(name=name, **info_name)
            else:
                raise TypeError(info_name)

            trfs_for_name = [
                trf for trf in sample_transformations if any([kwargs["apply_to"] == name for kwargs in trf.values()])
            ]
            info.transformations += trfs_for_name
            if "_repeat" in name:
                name, _ = name.split("_repeat")

            self.infos[name] = info

        if isinstance(indices, list):
            assert all(isinstance(i, int) for i in indices)
            self.indices = indices
        elif isinstance(indices, int):
            self.indices = [indices]
        elif isinstance(indices, str):
            self.indices = indice_string_to_list(indices)
        elif indices is None:
            self.indices = None
        else:
            raise NotImplementedError(indices)


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
            self._dataset = None

    def __len__(self):
        return len(self.dataset)


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
            assert len(self._data_setup.dataset), self._data_setup.groups
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
    seed: Optional[int] = None
    log: LogSetup
    log_class: Type[LogSetup]
    batch_multiplier: int = 1

    def __init__(
        self,
        *,
        name: str,
        data: List[Dict[str, Any]],
        sampler: Dict[str, Any],
        metrics: List[Dict[str, Any]],
        log: Dict[str, Any],
        batch_preprocessing: List[Dict[str, Dict[str, Any]]],
        batch_preprocessing_in_step: List[Dict[str, Dict[str, Any]]],
        batch_postprocessing: List[Dict[str, Dict[str, Any]]],
        model: LnetModel,
        log_path: Path,
        tensors_to_save: Optional[Sequence[str]] = tuple(),
        post_runs: Sequence[str] = tuple(),
    ):
        self.name = name
        self.log_path = log_path / name / "init"
        self.run_count = 0
        self.tensors_to_save = tensors_to_save
        self.metrics = metrics
        self._data_loader: Optional[torch.utils.data.DataLoader] = None
        self._engine: Optional[ignite.engine.Engine] = None
        self._epoch_length: Optional[int] = None
        self.model = model
        batch_preprocessing_instances: List[Transform] = [
            getattr(hylfm.transformations, name)(**kwargs)
            for trf in batch_preprocessing
            for name, kwargs in trf.items()
        ]
        self.batch_preprocessing = ComposedTransformation(*batch_preprocessing_instances)
        batch_preprocessing_in_step_instances: List[Transform] = [
            getattr(hylfm.transformations, name)(**kwargs)
            for trf in batch_preprocessing_in_step
            for name, kwargs in trf.items()
        ]
        self.batch_preprocessing_in_step = ComposedTransformation(*batch_preprocessing_in_step_instances)
        batch_postprocessing_instances: List[Transform] = [
            getattr(hylfm.transformations, name)(**kwargs)
            for trf in batch_postprocessing
            for name, kwargs in trf.items()
        ]
        self.batch_postprocessing = ComposedTransformation(*batch_postprocessing_instances)
        self.data: DataSetup = DataSetup([DatasetGroupSetup(**d) for d in data])
        self.sampler: SamplerSetup = SamplerSetup(**sampler, _data_setup=self.data)
        self.log = self.log_class(stage=self, **log)
        self.post_runs = [partial(getattr(post_run, name), **kwargs) for pr in post_runs for name, kwargs in pr.items()]

    @property
    def run_count(self):
        return self.__run_count

    @run_count.setter
    def run_count(self, rc: int):
        self.__run_count = rc
        if self.log_path.exists():
            delete_empty_dirs(self.log_path)
        self.log_path = self.log_path.parent / f"run{self.run_count:03}"
        self.log_path.mkdir(parents=True)

    @property
    def epoch_length(self):
        if self._epoch_length is None:
            self._epoch_length = len(self.data_loader)

        return self._epoch_length

    @property
    def data_loader(self) -> torch.utils.data.DataLoader:
        if self._data_loader is None:
            self._data_loader = torch.utils.data.DataLoader(
                dataset=self.data.dataset,
                batch_sampler=self.sampler.batch_sampler,
                collate_fn=get_collate_fn(batch_transformation=self.batch_preprocessing),
                num_workers=settings.num_workers_train_data_loader,
                pin_memory=settings.pin_memory,
            )

        return self._data_loader

    def start_engine(self, engine: ignite.engine.Engine):
        engine.state.compute_time = 0.0
        engine.state.stage = self
        engine.state.model = self.model
        engine.state.batch_postprocessing = self.batch_postprocessing

    def log_compute_time(self, engine: ignite.engine.Engine):
        mins, secs = divmod(engine.state.compute_time / max(1, engine.state.iteration), 60)
        msecs = (secs % 1) * 1000
        hours, mins = divmod(mins, 60)
        engine.logger.info(
            "%s ran %d iterations (batch sizes %s) in %.2f s with avg compute time per iteration %02d:%02d:%02d:%03d h:m:s:ms",
            self.name,
            engine.state.iteration,
            [g.batch_size for g in engine.state.stage.data.groups],
            engine.state.compute_time,
            hours,
            mins,
            secs,
            msecs,
        )

    def _setup_engine(self):
        self.engine.add_event_handler(ignite.engine.Events.STARTED, self.start_engine)
        self.engine.add_event_handler(ignite.engine.Events.COMPLETED, self.log_compute_time)
        self._attach_metrics()

    def _attach_metrics(self):
        metric_instances = [get_metric(**kwargs) for kwargs in self.metrics]
        for metric in metric_instances:
            metric.attach(self.engine, "")
            if any([lgr.scalars_every for lgr in self.log.loggers.values()]):
                self.engine.add_event_handler(ignite.engine.Events.ITERATION_COMPLETED, metric.completed, "")

    @property
    def engine(self):
        if self._engine is None:
            self._engine = ignite.engine.Engine(self.step_function)
            self._setup_engine()
            self.log.register_callbacks(engine=self._engine)

        return self._engine

    def setup(self):
        return self.engine, self.data_loader, self.max_epochs, self.epoch_length, self.seed

    def run(self):
        state = self.engine.run(
            data=self.data_loader,
            max_epochs=self.max_epochs,
            epoch_length=self.epoch_length - (self.epoch_length % self.batch_multiplier),
            seed=self.seed,
        )
        for post_run in self.post_runs:
            post_run(stage=self)

        self.run_count += 1
        return state

    def shutdown(self):
        self.log.shutdown()
        self.data.shutdown()


class EvalStage(Stage):
    step_function = staticmethod(inference_step)
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
        metrics: List[Dict[str, Any]],
        score_metric: str,
        **super_kwargs,
    ):
        super().__init__(metrics=metrics, **super_kwargs)
        self.period = Period(**period)
        self.patience = patience
        self.train_stage = train_stage
        self.score_metric = score_metric


@dataclass
class CriterionSetup:
    name: str
    kwargs: Dict[str, Any]
    _class: Type[LossOnTensors] = field(init=False)
    tensor_names: Dict[str, str]

    def __post_init__(self):
        self._class = getattr(hylfm.losses, self.name)
        self.name = camel_to_snake(self.name)
        assert "engine" not in self.kwargs
        assert "tensor_names" not in self.kwargs

    def get_criterion(self, *, engine: ignite.engine.Engine):
        sig = signature(self._class)
        kwargs = dict(self.kwargs)
        if "engine" in sig.parameters:
            kwargs["engine"] = engine

        return self._class(tensor_names=self.tensor_names, **kwargs)


@dataclass
class OptimizerSetup:
    name: str
    kwargs: Dict[str, Any]

    def __post_init__(self):
        self._class = getattr(hylfm.optimizers, self.name)
        assert "engine" not in self.kwargs
        assert "parameters" not in self.kwargs

    def get_optimizer(self, *, engine: ignite.engine.Engine):
        sig = signature(self._class)
        kwargs = dict(self.kwargs)
        if "engine" in sig.parameters:
            kwargs["engine"] = engine

        opt = self._class(engine.state.model.parameters(), **kwargs)
        opt.zero_grad()
        return opt


class TrainStage(Stage):
    log: TrainLogSetup
    log_class = TrainLogSetup
    step_function = staticmethod(training_step)

    def __init__(
        self,
        max_epochs: int,
        validate: Dict[str, Any],
        criterion: Dict[str, Dict[str, Any]],
        optimizer: Dict[str, Dict[str, Any]],
        model: Callable,
        log_path: Path,
        batch_multiplier: int = 1,
        **super_kwargs,
    ):
        super().__init__(log_path=log_path, model=model, **super_kwargs)
        self.max_epochs = max_epochs
        assert batch_multiplier > 0, batch_multiplier
        self.batch_multiplier = batch_multiplier
        self.criterion_setup = CriterionSetup(**criterion)
        self.optimizer_setup = OptimizerSetup(**optimizer)
        self.validate = ValidateStage(
            name="validate_" + self.name, train_stage=self, model=model, log_path=log_path, **validate
        )

    def _setup_engine(self):
        super()._setup_engine()
        running_loss = ignite.metrics.RunningAverage(output_transform=lambda out: out[self.criterion_setup.name])
        running_loss.attach(self.engine, f"{self.criterion_setup.name}-running")

        checkpointer = ignite.handlers.ModelCheckpoint(
            (self.log_path / "checkpoints").as_posix(),
            "v1",
            score_function=lambda e: e.state.validation_score,
            score_name=self.validate.score_metric,
            n_saved=self.log.save_n_checkpoints,
            create_dir=True,
            global_step_transform=global_step_from_engine(self.engine),
        )

        early_stopper = ignite.handlers.EarlyStopping(
            patience=self.validate.patience, score_function=lambda e: e.state.validation_score, trainer=self.engine
        )

        if self.validate.period.unit == PeriodUnit.epoch:
            event = ignite.engine.Events.EPOCH_COMPLETED
        elif self.validate.period.unit == PeriodUnit.iteration:
            event = ignite.engine.Events.ITERATION_COMPLETED
        else:
            raise NotImplementedError(self.validate.period.unit)

        @self.engine.on(event(every=self.validate.period.value))
        def validate(e: ignite.engine.Engine):
            validation_state = self.validate.run()
            metric_value: MetricValue = validation_state.metrics.get(self.validate.score_metric)
            validation_score = metric_value.value
            if not metric_value.higher_is_better:
                validation_score *= -1

            e.state.validation_score = validation_score
            checkpointer(
                e, {"model": e.state.model, "optimizer": e.state.optimizer, "criterion": e.state.criterion, "engine": e}
            )
            early_stopper(e)

        @self.engine.on(ignite.engine.Events.COMPLETED)
        def load_best_model(e: ignite.engine.Engine):
            best_score, best_checkpoint = checkpointer._saved[-1]
            assert best_score == max([item.priority for item in checkpointer._saved])
            model: torch.nn.Module = e.state.model
            model.load_state_dict(
                state_dict=torch.load(Path(checkpointer.save_handler.dirname) / best_checkpoint)["model"]
            )

    def start_engine(self, engine: ignite.engine.Engine):
        super().start_engine(engine)
        engine.state.optimizer = self.optimizer_setup.get_optimizer(engine=engine)
        engine.state.criterion = self.criterion_setup.get_criterion(engine=engine)
        engine.state.criterion_name = self.criterion_setup.name

    def shutdown(self):
        self.validate.shutdown()
        super().shutdown()

    def setup(self):
        self.validate.setup()
        super().setup()


class Setup:
    def __init__(
        self,
        *,
        config_path: Path,
        precision: str,
        device: Union[int, str] = 0,
        model: Dict[str, Any],
        stages: List[Dict[str, Any]],
        log_path: Optional[Union[Path, str]] = None,
        checkpoint: Optional[Union[Path, str]] = None,
        toolbox: Optional[dict] = None,
    ):
        # update config with generated and passed arguments

        if checkpoint is not None:
            checkpoint = checkpoint.as_posix()
            star_pos = checkpoint.find("*")
            if star_pos != -1:
                valid_path_until = checkpoint[:star_pos].rfind("/")
                checkpoint_dir = Path(checkpoint[:valid_path_until])
                assert checkpoint_dir.exists(), checkpoint_dir
                glob_expr = checkpoint[valid_path_until + 1 :]
                checkpoint = max([(float(c.stem.split("=")[1]), c) for c in checkpoint_dir.glob(glob_expr)])[1]

            checkpoint = Path(checkpoint)
            assert checkpoint.exists(), checkpoint

        model_checkpoint = model.get("checkpoint", None) or checkpoint
        model["checkpoint"] = None if model_checkpoint is None else str(model_checkpoint)

        assert all([len(stage) == 1 for stage in stages]), "invalid stage config"
        test_individually = [stage for stage in stages if list(stage.keys())[0] == "test_individually"]
        stages = [stage for stage in stages if list(stage.keys())[0] != "test_individually"]
        for ti in test_individually:
            assert len(ti) == 1, ti
            ti = ti["test_individually"]
            for idat in ti.pop("datasets"):
                stage = copy.deepcopy(ti)
                stage["data"][0]["datasets"] = [idat]
                stages.append({idat["tensors"]["lf"]: stage})

        self.config_path = config_path
        self.log_path = self.get_log_path(config_path=config_path) if log_path is None else Path(log_path)
        logger.info("log_dir: %s", self.log_path)

        self.config = self.load_subconfig_yaml(
            {
                "config_path": str(config_path),
                "checkpoint": None if checkpoint is None else str(checkpoint),
                "precision": precision,
                "device": device,
                "model": model,
                "stages": stages,
                "log_dir": str(log_path),
            }
        )

    def load_subconfig_yaml(self, part: Any, subconfig_dir: Path = settings.configs_dir / "subconfigs"):
        if isinstance(part, list):
            return [self.load_subconfig_yaml(p) for p in part]
        elif isinstance(part, dict):
            return {k: self.load_subconfig_yaml(v, subconfig_dir.parent / k) for k, v in part.items()}
        elif isinstance(part, str) and part.endswith(".yml"):
            if (subconfig_dir / part).exists():
                return yaml.load(subconfig_dir / part)

        return part

    @classmethod
    def from_yaml(cls, yaml_path: Path, **overwrite_kwargs) -> "Setup":
        config = yaml.load(yaml_path)
        config.update(overwrite_kwargs)
        return cls(**config, config_path=yaml_path)

    @staticmethod
    def get_log_path(config_path: Path, *, root: Optional[Path] = None) -> Path:
        config_path = config_path.resolve()
        assert settings.configs_dir in config_path.parents, (settings.configs_dir, config_path)
        log_sub_dir = config_path.relative_to(settings.configs_dir).with_suffix("")
        return (root or settings.log_dir) / log_sub_dir / datetime.now().strftime("%y-%m-%d_%H-%M-%S")

    def get_scaling(self, ipt_shape: Optional[Tuple[int, int]] = None) -> Tuple[float, float]:
        return self.model.get_scaling(ipt_shape)

    def setup(self) -> Path:
        self.log_path.mkdir(parents=True, exist_ok=False)
        # save original config as template
        shutil.copy(str(self.config_path), str(self.log_path / "template.yaml"))
        # save actual config to log path
        yaml.dump(self.config, self.log_path / "config.yaml")

        self.dtype: torch.dtype = getattr(torch, self.config["precision"])
        assert isinstance(self.dtype, torch.dtype)
        if isinstance(self.config["device"], int) or "cuda" in self.config["device"]:
            cuda_device_count = torch.cuda.device_count()
            if cuda_device_count == 0:
                raise RuntimeError("no CUDA devices available!")
            elif cuda_device_count > 1:
                raise RuntimeError("too many CUDA devices available! (limit to one)")

        self.device = torch.device(self.config["device"])
        self.model_setup = ModelSetup(**self.config["model"])
        self.model: LnetModel = self.model_setup.get_model(device=self.device, dtype=self.dtype)
        self.stages = [
            {
                (
                    TrainStage(**kwargs, name=name, model=self.model, log_path=self.log_path)
                    if "optimizer" in kwargs
                    else EvalStage(**kwargs, name=name, model=self.model, log_path=self.log_path)
                )
                for name, kwargs in stage.items()
            }
            for stage in self.config["stages"]
        ]

        for parallel_stages in self.stages:
            for stage in parallel_stages:
                logger.info("setup stage: %s", stage.name)
                stage.setup()

        return self.log_path

    def run(self) -> Path:
        self.setup()
        try:
            for parallel_stages in self.stages:
                for stage in parallel_stages:
                    logger.info("starting stage: %s", stage.name)
                    state = stage.run()
                    stage.shutdown()
        finally:
            self.shutdown()
            delete_empty_dirs(self.log_path)

        return self.log_path

    def shutdown(self):
        [stage.shutdown() for parallel_stages in self.stages for stage in parallel_stages]
