import logging
from enum import Enum

from dataclasses import dataclass, field, InitVar
from typing import Optional, Dict, Any, Union, List, Generator

from inferno.io.transform import Transform
from lnet import models
from torch.utils.data import DataLoader, ConcatDataset, Subset, RandomSampler, SequentialSampler

from lnet.config.model import ModelConfig
from lnet.config.utils import get_trfs_and_their_names
from lnet.config.dataset import beads, platy, fish, NamedDatasetInfo
from lnet.datasets import N5Dataset
from lnet.transforms import known_transforms, randomly_shape_changing_transforms
from lnet.utils.batch_sampler import NoCrossBatchSampler
from lnet.utils.transforms import EdgeCrop

logger = logging.getLogger(__name__)


class DataCategory(Enum):
    train = "train_data"
    valid = "valid_data"
    test = "test_data"


@dataclass
class DataConfigEntry:
    model_config: InitVar[ModelConfig]

    name: str
    batch_size: int
    indices: Optional[List[int]] = None

    transforms: List[Union[Generator[Transform, None, None], Transform]] = None
    transform_configs: List[Union[str, Dict[str, Union[str, Dict[str, Any]]]]] = None

    transform_names: List[str] = field(init=False)
    info: NamedDatasetInfo = field(init=False)

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
                concatenated_indices += DataConfigEntry.range_or_single_index_to_list(part)

            return concatenated_indices

    def __post_init__(self, model_config: ModelConfig):
        self.transforms, self.transform_names = get_trfs_and_their_names(
            model_config=model_config, transforms=self.transforms, conf=self.transform_configs
        )
        self.transforms.append(known_transforms["Cast"](model_config=model_config, kwargs={}))
        if hasattr(model_config.Model, "get_shrinkage"):
            self.transforms.append(EdgeCrop(model_config.Model.get_shrinkage(), apply_to=[1]))

        if self.batch_size > 1:
            for t in randomly_shape_changing_transforms:
                if t in self.transform_names:
                    raise ValueError(
                        f"{self.name}: batch size={self.batch_size} > 1 is not compatible with transforms that "
                        f"randomly change shape (here: {t})"
                    )

        self.info = getattr(beads, self.name, None) or getattr(fish, self.name, None) or getattr(platy, self.name, None)
        if self.info is None:
            raise NotImplementedError(f"could not find named dataset info `{self.name}`")

    @classmethod
    def load(
        cls: "DataConfigEntry",
        model_config: ModelConfig,
        indices: Optional[Union[str, int]],
        transforms: List[Union[str, Dict[str, Union[str, Dict[str, Any]]]]],
        **kwargs,
    ) -> "DataConfigEntry":
        return cls(
            model_config=model_config,
            indices=cls.indice_string_to_list(indices),
            transform_configs=transforms,  # note: 'transform_configs' is called 'transforms' in yaml!
            **kwargs,
        )


@dataclass
class DataConfig:
    model_config: InitVar[ModelConfig]

    category: DataCategory
    entries: List[DataConfigEntry]

    data_loader: DataLoader = field(init=False)

    z_out: int = field(init=False)

    def __post_init__(self, model_config: ModelConfig):

        datasets = [
            N5Dataset(
                info=entry.info,
                scaling=getattr(models, model_config.name).get_scaling(),
                interpolation_order=3,
                save=True,
                transforms=entry.transforms,
            )
            for entry in self.entries
        ]

        # todo: move to project specific code:
        z_outs = [ds.z_out for ds in datasets]
        self.z_out = z_outs[0]
        assert all(zo == self.z_out for zo in z_outs), z_outs

        concat_dataset = ConcatDataset(
            [ds if entry.indices is None else Subset(ds, entry.indices) for ds, entry in zip(datasets, self.entries)]
        )

        self.data_loader = DataLoader(
            concat_dataset,
            batch_sampler=NoCrossBatchSampler(
                concat_dataset=concat_dataset,
                sampler_class=RandomSampler if self.category == DataCategory.train else SequentialSampler,
                batch_sizes=[e.batch_size for e in self.entries],
                drop_last=False,
            ),
            pin_memory=True,
            num_workers=8,
        )

    @classmethod
    def load(
        cls: "DataConfig",
        model_config: ModelConfig,
        category: DataCategory,
        entries: Dict[str, Dict[str, Any]],
        default_batch_size: Optional[int] = None,
        default_transforms: Optional[List[Union[str, Dict[str, Union[str, Dict[str, Any]]]]]] = None,
    ) -> "DataConfig":
        return cls(
            category=category,
            entries=[
                DataConfigEntry.load(
                    model_config=model_config,
                    name=name,
                    batch_size=kwargs.pop("batch_size", default_batch_size),
                    transforms=kwargs.pop("transforms", default_transforms),
                    **kwargs,
                )
                for name, kwargs in entries
            ],
        )
