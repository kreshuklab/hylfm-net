from __future__ import annotations

import logging
import shutil
import typing
from dataclasses import dataclass, field
from datetime import datetime
from importlib import import_module
from inspect import signature
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type, Union

import ignite
import torch.utils.data
import yaml

import lnet.criteria
import lnet.log
import lnet.metrics
import lnet.optimizers
import lnet.registration
import lnet.transformations
from lnet import settings
from lnet.datasets import ConcatDataset, N5CachedDataset, ZipDataset, ZipSubset, get_collate_fn, get_dataset_from_info
from lnet.datasets.base import TensorInfo
from lnet.models import LnetModel
from lnet.setup._utils import indice_string_to_list
from lnet.step import inference_step, training_step
from lnet.transformations import ComposedTransform, Transform
from lnet.utils import Period, PeriodUnit
from lnet.utils.batch_sampler import NoCrossBatchSampler

logger = logging.getLogger(__name__)


class ModelSetup:
    def __init__(
        self,
        name: str,
        kwargs: Dict[str, Any],
        checkpoint: Optional[Union[str, Tuple[str, str]]] = None,
        partial_weights: bool = False,
    ):
        self.name = name
        self.kwargs = kwargs
        self.checkpoint = None if checkpoint is None else getattr(settings.data_roots, checkpoint[0]) / checkpoint[1]
        if self.checkpoint is not None:
            assert len(checkpoint) == 2, f"expected root, file_path, but got: {checkpoint}"
            assert self.checkpoint.exists()
        self.partial_weights = partial_weights

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


class PerLoggerSetup:
    def __init__(
        self,
        name: str,
        stage: Stage,
        scalars_every: Dict[str, Union[int, str]],
        tensors_every: Dict[str, Union[int, str]],
        tensor_names: Optional[typing.Set[str]] = None,
    ):
        self.backend: lnet.log.BaseLogger = getattr(lnet.log, name)(stage=stage, tensor_names=tensor_names)
        self.scalars_every = Period(**scalars_every)
        self.tensors_every = Period(**tensors_every)

    def register_callbacks(self, engine: ignite.engine.Engine):
        self.backend.register_callbacks(
            engine=engine, scalars_every=self.scalars_every, tensors_every=self.tensors_every
        )

    def shutdown(self):
        self.backend.shutdown()


class LogSetup:
    def __init__(self, *, stage: Stage, **loggers: dict):
        self.loggers = {name: PerLoggerSetup(name=name, stage=stage, **lgr) for name, lgr in loggers.items()}

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
        batch_size: int,
        interpolation_order: int,
        datasets: List[Dict[str, Any]],
        sample_preprocessing: List[Dict[str, Dict[str, Any]]],
        ls_affine_transform_class: Optional[str] = None,
    ):
        self.batch_size = batch_size
        self.interpolation_order = interpolation_order

        self.ls_affine_transform_class = (
            None if ls_affine_transform_class is None else getattr(lnet.registration, ls_affine_transform_class)
        )
        sample_prepr_trf_instances: List[Transform] = [
            getattr(lnet.transformations, name)(**kwargs) for trf in sample_preprocessing for name, kwargs in trf.items()
        ]
        self.sample_preprocessing = ComposedTransform(*sample_prepr_trf_instances)
        self._dataset: Optional[ConcatDataset] = None
        self.dataset_setups: List[DatasetSetup] = [DatasetSetup(**kwargs) for kwargs in datasets]

    def get_individual_dataset(self, dss: DatasetSetup) -> torch.utils.data.Dataset:
        ds = ZipDataset(
            {name: N5CachedDataset(get_dataset_from_info(dsinfo)) for name, dsinfo in dss.infos.items()},
            transformation=self.sample_preprocessing,
        )

        if dss.indices is None:
            return ds
        else:
            return ZipSubset(ds, dss.indices, dss.z_crop)

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
        interpolation_order: int,
        indices: Optional[Union[str, int, List[int]]] = None,
        z_crop: Optional[Tuple[int, int]] = None,
        sample_transformations: Sequence[Dict[str, Dict[str, Any]]] = tuple(),
    ):
        self.infos = {}
        for name, info_name in tensors.items():
            if isinstance(info_name, str):
                info_module_name, info_name = info_name.split(".")
                info_module = import_module("." + info_module_name, "lnet.datasets")
                info = getattr(info_module, info_name)
            elif isinstance(info_name, dict):
                info = TensorInfo(**info_name)

            info.transformations += list(sample_transformations)
            self.infos[name] = info
                
        self.interpolation_order = interpolation_order

        if isinstance(indices, list):
            assert all(isinstance(i, int) for i in indices)
            self.indices = indices
        elif isinstance(indices, int):
            self.indices = [indices]
        elif isinstance(indices, str):
            self.indices = indice_string_to_list(indices)
        else:
            raise NotImplementedError(indices)

        assert z_crop is None or len(z_crop) == 2
        self.z_crop = z_crop


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

    def __init__(
        self,
        *,
        name: str,
        data: List[Dict[str, Any]],
        sampler: Dict[str, Any],
        metrics: Dict[str, Dict[str, Any]],
        log: Dict[str, Any],
        batch_preprocessing: List[Dict[str, Dict[str, Any]]],
        batch_postprocessing: List[Dict[str, Dict[str, Any]]],
        model: LnetModel,
        log_path: Path,
        tensors_to_save: Optional[Sequence[str]] = tuple(),
    ):
        self.name = name
        self.tensors_to_save = tensors_to_save
        self.metrics = metrics
        self._data_loader: Optional[torch.utils.data.DataLoader] = None
        self._engine: Optional[ignite.engine.Engine] = None
        self._epoch_length: Optional[int] = None
        self.model = model
        self.log_path = log_path / name
        batch_preprocessing_instances: List[Transform] = [
            getattr(lnet.transformations, name)(**kwargs) for trf in batch_preprocessing for name, kwargs in trf.items()
        ]
        self.batch_preprocessing = ComposedTransform(*batch_preprocessing_instances)
        batch_postprocessing_instances: List[Transform] = [
            getattr(lnet.transformations, name)(**kwargs)
            for trf in batch_postprocessing
            for name, kwargs in trf.items()
        ]
        self.batch_postprocessing = ComposedTransform(*batch_postprocessing_instances)
        self.data: DataSetup = DataSetup([DatasetGroupSetup(**d) for d in data])
        self.sampler: SamplerSetup = SamplerSetup(**sampler, _data_setup=self.data)
        self.log = self.log_class(stage=self, **log)

    @property
    def epoch_length(self):
        if self._epoch_length is None:
            self._epoch_length = len(self.data)

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
            "%s run on %d mini-batches completed in %.2f s with avg compute time %02d:%02d:%02d:%03d",
            self.name,
            len(engine.state.dataloader),
            engine.state.compute_time,
            hours,
            mins,
            secs,
            msecs,
        )

    def setup_engine(self, engine: ignite.engine.Engine):
        engine.add_event_handler(ignite.engine.Events.STARTED, self.start_engine)
        engine.add_event_handler(ignite.engine.Events.COMPLETED, self.log_compute_time)

        initialized_metrics = {}
        for metric_name, kwargs in self.metrics.items():
            kwargs = dict(kwargs)
            metric_class = getattr(lnet.metrics, metric_name, None)
            if metric_class is None:
                if (
                    isinstance(self, TrainStage)
                    and metric_name == self.criterion_setup.name
                    and {k: v for k, v in kwargs.items() if k != "tensor_names"} == self.criterion_setup.kwargs
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
                        postfix = kwargs.pop("postfix", "")
                        metric_name += postfix
                        tensor_names = kwargs.pop("tensor_names")
                        criterion = criterion_class(**kwargs)
                        criterion.eval()
                        arg_names = [tensor_names[an] for an in ("y_pred", "y", "kwargs") if an in tensor_names]
                        metric = ignite.metrics.Loss(criterion, lambda tensors: [tensors[an] for an in arg_names])
            else:
                try:
                    metric = metric_class(
                        output_transform=lnet.metrics.get_output_transform(kwargs.pop("tensor_names")), **kwargs
                    )
                except Exception:
                    logger.error("Cannot init %s", metric_name)
                    raise

            metric.attach(engine, metric_name)

            # def shutdown_data_setup(engine: ignite.engine.Engine, data_setup: DataSetup = self.data):
            #     data_setup.shutdown()
            #
            # self._engine.add_event_handler(ignite.engine.Events.COMPLETED, shutdown_data_setup)

    @property
    def engine(self):
        if self._engine is None:
            self.log_path.mkdir(parents=True, exist_ok=False)
            self._engine = ignite.engine.Engine(self.step_function)
            self.setup_engine(self._engine)
            self.log.register_callbacks(engine=self._engine)

        return self._engine

    def run(self):
        return self.engine.run(
            data=self.data_loader, max_epochs=self.max_epochs, epoch_length=self.epoch_length, seed=self.seed
        )

    def shutdown(self):
        self.log.shutdown()
        self.data.shutdown()


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
        assert score_metric in metrics or score_metric.startswith("-") and score_metric[1:] in metrics
        self.negate_score_metric = score_metric.startswith("-")
        self.score_metric = score_metric[int(self.negate_score_metric) :]


@dataclass
class CriterionSetup:
    name: str
    kwargs: Dict[str, Any]
    _class: Type[torch.nn.Module] = field(init=False)
    tensor_names: Dict[str, str]
    postfix: str = ""

    def __post_init__(self):
        self._class = getattr(lnet.criteria, self.name)
        self.name += self.postfix
        assert "engine" not in self.kwargs
        assert "tensor_names" not in self.kwargs

    def get_criterion(self, *, engine: ignite.engine.Engine):
        sig = signature(self._class)
        kwargs = dict(self.kwargs)
        if "engine" in sig.parameters:
            kwargs["engine"] = engine

        return lnet.criteria.CriterionWrapper(
            tensor_names=self.tensor_names, criterion_class=self._class, postfix=self.postfix, **kwargs
        )


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
        model: Callable,
        log_path: Path,
        **super_kwargs,
    ):
        super().__init__(log_path=log_path, model=model, **super_kwargs)
        self.max_num_epochs = max_num_epochs
        self.criterion_setup = CriterionSetup(**criterion)
        self.optimizer_setup = OptimizerSetup(**optimizer)
        self.validate = ValidateStage(name="validate", train_stage=self, model=model, log_path=log_path, **validate)

    def setup_engine(self, engine: ignite.engine.Engine):
        super().setup_engine(engine)
        running_loss = ignite.metrics.RunningAverage(output_transform=lambda out: out[self.criterion_setup.name])
        running_loss.attach(engine, f"{self.criterion_setup.name}-running")

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

        if self.validate.period.unit == PeriodUnit.epoch:
            event = ignite.engine.Events.EPOCH_COMPLETED
        elif self.validate.period.unit == PeriodUnit.iteration:
            event = ignite.engine.Events.ITERATION_COMPLETED
        else:
            raise NotImplementedError

        @engine.on(event(every=self.validate.period.value))
        def validate(e: ignite.engine.Engine):
            validation_state = self.validate.run()
            validation_score = validation_state.metrics.get(self.validate.score_metric)
            if self.validate.negate_score_metric:
                validation_score *= -1

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
        toolbox: Optional[dict] = None,
    ):
        self.dtype: torch.dtype = getattr(torch, precision)
        assert isinstance(self.dtype, torch.dtype)
        self.nnum = nnum
        self.z_out = z_out
        self.config_path = config_path
        self.data_cache_path = self.get_data_cache_path() if data_cache_path is None else Path(data_cache_path)
        self.log_path = self.get_log_path() if log_path is None else Path(log_path)
        if isinstance(device, int) or "cuda" in device:
            cuda_device_count = torch.cuda.device_count()
            if cuda_device_count == 0:
                raise RuntimeError("no CUDA devices available!")
            elif cuda_device_count > 1:
                raise RuntimeError("too many CUDA devices available! (limit to one)")

        self.device = torch.device(device)
        self.model_setup = ModelSetup(**model)
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
            for stage in stages
        ]

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "Setup":
        with yaml_path.open() as f:
            config = yaml.safe_load(f)

        return cls(**config, config_path=yaml_path)

    def get_data_cache_path(self) -> Path:
        data_cache_path = Path(__file__).parent / "../../data"
        data_cache_path.mkdir(exist_ok=True)

        return data_cache_path

    def get_log_path(self) -> Path:
        log_sub_dir: List[str] = self.config_path.with_suffix("").resolve().as_posix().split("/experiment_configs/")
        assert len(log_sub_dir) == 2, log_sub_dir
        log_sub_dir: str = log_sub_dir[1]
        log_path = settings.log_path / log_sub_dir / datetime.now().strftime("%y-%m-%d_%H-%M-%S")

        logger.info("log_path: %s", log_path)
        log_path.mkdir(parents=True, exist_ok=False)

        # try:
        #     commit_hash = pbs3.git("rev-parse", "--verify", "HEAD").stdout
        # except pbs3.CommandNotFound:
        #     commit_hash = subprocess.run(
        #         ["git", "rev-parse", "--verify", "HEAD"], capture_output=True, text=True
        #     ).stdout
        # (log_path / "full_commit_hash.txt").write_text(commit_hash)
        shutil.copy(self.config_path.as_posix(), (log_path / "config.yaml").as_posix())

        return log_path

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
