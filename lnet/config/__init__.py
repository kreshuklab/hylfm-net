import logging
from dataclasses import InitVar, dataclass, field
from enum import Enum
from importlib import import_module
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy
import torch.utils.data

import lnet.registration
import lnet.transforms
from lnet.datasets import ConcatDataset, N5ChunkAsSampleDataset, NamedDatasetInfo
from lnet.models import LnetModel
from lnet.transforms.base import ComposedTransform, Transform
from lnet.utils.batch_sampler import NoCrossBatchSampler
from ._utils import enforce_types

logger = logging.getLogger(__name__)


class PeriodUnit(Enum):
    epoch = "epoch"
    iteration = "iteration"


@enforce_types
@dataclass
class Period:
    value: int
    unit: str

    def __post_init__(self):
        self.unit: PeriodUnit = PeriodUnit(self.unit)


@enforce_types
@dataclass
class ModelConfig:
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

                    self.logger.warning("!!!State from checkpoint does not match!!!")

                    # load partial state
                    before = "size mismatch for "
                    after = ": copying a param with shape"
                    for line in str(e).split("\n")[1:]:
                        idx_before = line.find(before)
                        idx_after = line.find(after)
                        if idx_before == -1 or idx_after == -1:
                            self.logger.warning("Didn't understand 'load_state_dict' exception line: %s", line)
                        else:
                            state.pop(line[idx_before + len(before) : idx_after])
        return model


@enforce_types
@dataclass
class LogConfig:
    log_scalars_period: Dict[str, Union[int, str]]
    log_images_period: Dict[str, Union[int, str]]
    log_bead_precision_recall: bool = False
    log_bead_precision_recall_threshold: float = 5.0
    save_n_checkpoints: int = 1

    def __post_init__(self):
        self.log_scalars_period: Period = Period(**self.log_scalars_period)
        self.log_images_period: Period = Period(**self.log_images_period)


@enforce_types
@dataclass
class DatasetGroupConfig:
    batch_size: int
    datasets: Dict[str, Dict[str, Any]]
    interpolation_order: int
    transforms: List[Dict[str, Any]]
    _nnum: int
    _z_out: int
    _data_cache_path: Path
    _get_model_scaling: Callable
    ls_affine_transform_class: Optional[str] = None
    sample_transform: ComposedTransform = field(init=False)
    batch_transform: ComposedTransform = field(init=False)

    def __post_init__(self):
        trf_instances = []
        for trf_config in self.transforms:
            name = trf_config["name"]  # noqa
            try:
                kwargs = trf_config["kwargs"]  # noqa
                trf_class = getattr(lnet.transforms, name)
                trf_instances.append(trf_class(**kwargs))
            except Exception as e:
                raise ValueError(name) from e

        self.transforms: List[Transform] = [
            getattr(lnet.transforms, trf["name"])(**trf["kwargs"]) for trf in self.transforms  # noqa
        ]
        batch_transform_start = numpy.argmax([trf.randomly_changes_shape for trf in self.transforms])
        self.sample_transform = ComposedTransform(*self.transforms[:batch_transform_start])
        self.batch_transform = ComposedTransform(*self.transforms[batch_transform_start:])

        self.datasets: Dict[str, DatasetConfig] = {
            name: DatasetConfig(**kwargs, group_config=self, name=name) for name, kwargs in self.datasets.items()
        }
        self.ls_affine_transform_class = (
            None
            if self.ls_affine_transform_class is None
            else getattr(lnet.registration, self.ls_affine_transform_class)
        )

    def get_individual_dataset(self, name: str) -> N5ChunkAsSampleDataset:
        config = self.datasets[name]
        return N5ChunkAsSampleDataset(
            info=config.info,
            nnum=self._nnum,
            z_out=self._z_out,
            interpolation_order=self.interpolation_order,
            data_cache_path=self._data_cache_path,
            get_model_scaling=self.get_model_scaling,
            transform=self.sample_transform,
            ls_affine_transform_class=self.ls_affine_transform_class,
        )

    def get_dataset(self) -> ConcatDataset:
        return ConcatDataset(
            [self.get_individual_dataset(name) for name in self.datasets.keys()], transform=self.batch_transform
        )


@enforce_types
@dataclass
class DatasetConfig:
    name: str
    indices: Union[str, int]
    group_config: InitVar[DatasetGroupConfig]
    interpolation_order: Optional[int] = None
    transform: ComposedTransform = field(init=False)
    info: NamedDatasetInfo = field(init=False)

    def __post_init__(self, group_config: DatasetGroupConfig):
        self.interpolation_order = None if self.interpolation_order is None else group_config.interpolation_order
        self.transform = group_config.sample_transform
        info_module_name, info_name = self.name.split(".")
        info_module = import_module("." + info_module_name, "lnet.datasets")
        self.info = getattr(info_module, info_name)

    # @property
    # def dataset(self):
    #     return


class DataConfig:
    def __init__(self, dataset_groups: List[DatasetGroupConfig]):
        self.groups = dataset_groups
        self._dataset = None

    @property
    def dataset(self) -> ConcatDataset:
        if self._dataset is None:
            self._dataset = ConcatDataset([group.dataset for group in self.groups])

        return self._dataset

    @property
    def batch_sizes(self) -> List[int]:
        return [group.batch_size for group in self.groups]


@enforce_types
@dataclass
class TestConfig:
    data: List[Dict[str, Any]]
    _nnum: InitVar[int]
    _z_out: InitVar[int]
    _data_cache_path: InitVar[int]
    _get_model_scaling: InitVar[Callable]

    def __post_init__(self, _nnum: int, _z_out: int, _data_cache_path: Path, _get_model_scaling: Callable):
        data: List[Dict[str, Any]] = self.data  # noqa
        self.data: DataConfig = DataConfig(
            [
                DatasetGroupConfig(
                    **d,
                    _nnum=_nnum,
                    _z_out=_z_out,
                    _data_cache_path=_data_cache_path,
                    _get_model_scaling=_get_model_scaling
                )
                for d in data
            ]
        )


@enforce_types
@dataclass
class ValidateConfig:
    data: List[Dict[str, Any]]
    period: Dict[str, Union[int, str]]
    _nnum: InitVar[int]
    _z_out: InitVar[int]
    _data_cache_path: InitVar[int]
    _get_model_scaling: InitVar[Callable]

    def __post_init__(self, _nnum: int, _z_out: int, _data_cache_path: Path, _get_model_scaling: Callable):
        data: List[Dict[str, Any]] = self.data  # noqa
        self.data: DataConfig = DataConfig(
            [
                DatasetGroupConfig(
                    **d,
                    _nnum=_nnum,
                    _z_out=_z_out,
                    _data_cache_path=_data_cache_path,
                    _get_model_scaling=_get_model_scaling
                )
                for d in data
            ]
        )
        self.period: Period = Period(**self.period)


@enforce_types
@dataclass
class LossConfig:
    name: str
    kwargs: Dict[str, Any]


@enforce_types
@dataclass
class OptimizerConfig:
    name: str
    kwargs: Dict[str, Any]


@enforce_types
@dataclass
class SamplerConfig:
    strategy: str
    drop_last: bool

    data_config: InitVar[DataConfig]

    def __post_init__(self, data_config):
        base_sampler = getattr(torch.utils.data, self.strategy)
        self.sampler: NoCrossBatchSampler = NoCrossBatchSampler(
            data_config.dataset,
            sampler_class=base_sampler,
            batch_sizes=data_config.batch_sizes,
            drop_last=self.drop_last,
        )


@enforce_types
@dataclass
class TrainConfig:
    max_num_epochs: int
    score_metric: str
    patience: int
    log: Dict[str, Dict[str, Any]]
    validate: Dict[str, Dict[str, Any]]
    loss: Dict[str, Dict[str, Any]]
    optimizer: Dict[str, Dict[str, Any]]
    data: List[Dict[str, Any]]

    _nnum: InitVar[int]
    _z_out: InitVar[int]
    _data_cache_path: InitVar[Path]
    _get_model_scaling: InitVar[Callable]

    def __post_init__(self, _nnum: int, _z_out: int, _data_cache_path: Path, _get_model_scaling: Callable):
        self.log: LogConfig = LogConfig(**self.log)
        self.validate: ValidateConfig = ValidateConfig(
            **self.validate,
            _nnum=_nnum,
            _z_out=_z_out,
            _data_cache_path=_data_cache_path,
            _get_model_scaling=_get_model_scaling
        )
        self.loss: LossConfig = LossConfig(**self.loss)
        self.optimizer: OptimizerConfig = OptimizerConfig(**self.optimizer)
        data: List[Dict[str, Any]] = self.data  # noqa
        self.data: DataConfig = DataConfig(
            [
                DatasetGroupConfig(
                    **d,
                    _nnum=_nnum,
                    _z_out=_z_out,
                    _data_cache_path=_data_cache_path,
                    _get_model_scaling=_get_model_scaling
                )
                for d in data
            ]
        )


@enforce_types
@dataclass
class Config:
    data_cache_path: str
    precision: str
    nnum: int
    z_out: int
    model: Dict[str, Any]
    test: Dict[str, Any]
    train: Optional[Dict[str, Any]]

    dtype: torch.dtype = field(init=False)
    device: Union[int, str] = 0

    def __post_init__(self):
        self.dtype = getattr(torch, self.precision)
        if isinstance(self.device, int) or "cuda" in self.device:
            cuda_device_count = torch.cuda.device_count()
            if cuda_device_count == 0:
                raise RuntimeError("no CUDA devices available!")
            elif cuda_device_count > 1:
                raise RuntimeError("too many CUDA devices available! (limit to one)")

        self.device: torch.device = torch.device(self.device)  # noqa
        self.data_cache_path = Path(self.data_cache_path)
        self.model_config: ModelConfig = ModelConfig(**self.model)
        self.model: LnetModel = self.model_config.get_model(device=self.device, dtype=self.dtype)
        self.test: TestConfig = TestConfig(
            **self.test,
            _nnum=self.nnum,
            _z_out=self.z_out,
            _data_cache_path=self.data_cache_path,
            _get_model_scaling=self.model.get_scaling
        )
        self.train: Optional[TrainConfig] = None if self.train is None else TrainConfig(
            **self.train,
            _nnum=self.nnum,
            _z_out=self.z_out,
            _data_cache_path=self.data_cache_path,
            _get_model_scaling=self.model.get_scaling
        )
