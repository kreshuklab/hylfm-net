import logging
import sys
import warnings
from dataclasses import InitVar, dataclass, field
from enum import Enum
from typing import Any, Dict, Generator, List, Optional, Type, Union

from inferno.io.transform import Transform
from torch.utils.data import ConcatDataset, DataLoader, Dataset, RandomSampler, SequentialSampler, Subset

from lnet import registration
from lnet.config.utils import get_trfs_and_their_names
from lnet.datasets import (
    N5Dataset,
    NamedDatasetInfo,
    beads,
    fish,
    fish1_20191203,
    fish1_20191207,
    fish1_20191208,
    fish1_20191209,
    fish2_20191209,
    fish2_20191209_dynamic,
    fish3_20191209,
    nema,
    platy,
    tuesday_fish,
)
from lnet.registration import BDVTransform
from lnet.transforms import known_transforms, randomly_shape_changing_transforms
from lnet.utils.batch_sampler import NoCrossBatchSampler
from lnet.utils.transforms import EdgeCrop
from .model import ModelConfig

logger = logging.getLogger(__name__)


class DataCategory(Enum):
    train = "train_data"
    eval_train = "eval_train_data"
    valid = "valid_data"
    test = "test_data"


@dataclass
class DataConfigEntry:
    model_config: InitVar[ModelConfig]

    name: str
    batch_size: int
    indices: Optional[List[int]] = None
    save: bool = True
    interpolation_order: int = 3
    affine_transformation: Optional[str] = None

    transforms: List[Union[Generator[Transform, None, None], Transform]] = None
    transform_configs: List[Union[str, Dict[str, Union[str, Dict[str, Any]]]]] = None

    transform_names: List[str] = field(init=False)
    info: NamedDatasetInfo = field(init=False)
    AffineTransformation: Optional[Type[BDVTransform]] = field(init=False)

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
        if indice_string is None:
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
        if hasattr(model_config.model, "get_shrinkage"):
            self.transforms.append(EdgeCrop(apply_to=[1], crop_fn=model_config.model.get_shrinkage))

        if self.batch_size is None:
            raise ValueError(f"batch size not specified for {self.name}")

        if self.batch_size > 1:
            for t in randomly_shape_changing_transforms:
                if t in self.transform_names:
                    raise ValueError(
                        f"{self.name}: batch size={self.batch_size} > 1 is not compatible with transforms that "
                        f"randomly change shape (here: {t})"
                    )

        info_modules = {
            m.__name__.split(".")[-1]: m
            for m in [
                beads,
                platy,
                fish,
                nema,
                tuesday_fish,
                fish1_20191203,
                fish1_20191207,
                fish1_20191208,
                fish1_20191209,
                fish2_20191209,
                fish3_20191209,
                fish2_20191209_dynamic,
            ]
        }
        if "." in self.name:
            module_name, info_name = self.name.split(".")
            self.info = getattr(info_modules[module_name], info_name)
            self.info.description = module_name + "_" + self.info.description
        else:
            self.info = (
                getattr(beads, self.name, None)
                or getattr(fish, self.name, None)
                or getattr(platy, self.name, None)
                or getattr(nema, self.name)
            )
        if self.info is None:
            raise NotImplementedError(f"could not find named dataset info `{self.name}`")

        if self.info.length is not None and self.indices is not None and self.info.length <= max(self.indices):
            raise ValueError(f"{self.info.length} <= {max(self.indices)}")

        self.AffineTransformation = (
            self.info.DefaultAffineTransform
            if self.affine_transformation == "default"
            else None
            if self.affine_transformation is None
            else getattr(registration, self.affine_transformation)
        )

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

    datasets: List[Dataset] = field(init=False)
    concat_dataset: ConcatDataset = field(init=False)
    data_loader: DataLoader = field(init=False)

    z_out: Optional[int] = field(init=False)

    def __post_init__(self, model_config: ModelConfig):
        # check that the A01 hack works of identifying which transform to apply based on tensor shape
        x_shape_unique_to_affine_transform = {}
        for entry in self.entries:
            x_shape = ",".join(
                str(s)
                for s in (
                    model_config.nnum ** 2,
                    entry.info.x_shape[0] // model_config.nnum,
                    entry.info.x_shape[1] // model_config.nnum,
                )
            )
            at_name = x_shape_unique_to_affine_transform.get(x_shape, None)

            if at_name is None:
                at = entry.info.DefaultAffineTransform
                if at is not None:
                    x_shape_unique_to_affine_transform[x_shape] = at.__name__

                    a01_affine_transform_classes = model_config.kwargs.get("affine_transform_classes", None)
                    if a01_affine_transform_classes is not None:
                        if x_shape not in a01_affine_transform_classes:
                            warnings.warn(
                                f"x shape {x_shape} missing in model kwargs.affine_transformation_classes for entry {entry.name}"
                            )
                        elif a01_affine_transform_classes[x_shape] != at.__name__:
                            raise ValueError(
                                f"missmatch for kwargs:affine_transform_classes[x_shape={x_shape}]={a01_affine_transform_classes[x_shape]}!={at.__name__} for entry {entry.name}"
                            )
            else:
                if entry.info.DefaultAffineTransform is not None:
                    if at_name != entry.info.DefaultAffineTransform.__name__:
                        raise ValueError(f"x shape {x_shape} already associated with transform {at_name}")

        self.datasets: List[N5Dataset] = [
            N5Dataset(
                info=entry.info,
                scaling=None,
                z_out=model_config.z_out,
                interpolation_order=entry.interpolation_order,
                save=entry.save,
                transforms=entry.transforms,
                model_config=model_config,
                AffineTransformation=entry.AffineTransformation,
            )
            for entry in self.entries
        ]

        # todo: move to project specific code:
        z_outs = [ds.z_out for ds in self.datasets if ds.z_out is not None]
        if z_outs:
            self.z_out = z_outs[0]
            assert all(zo == self.z_out for zo in z_outs), z_outs
        else:
            self.z_out = None

        self.concat_dataset = ConcatDataset(
            [
                ds if entry.indices is None else Subset(ds, entry.indices)
                for ds, entry in zip(self.datasets, self.entries)
            ]
        )

        self.data_loader = DataLoader(
            self.concat_dataset,
            batch_sampler=NoCrossBatchSampler(
                concat_dataset=self.concat_dataset,
                sampler_class=RandomSampler if self.category == DataCategory.train else SequentialSampler,
                batch_sizes=[e.batch_size for e in self.entries],
                drop_last=False,
            ),
            pin_memory=True,
            num_workers=0 if sys.gettrace() is not None else 8,  # debug without worker threads
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
            model_config=model_config,
            category=category,
            entries=[
                DataConfigEntry.load(
                    model_config=model_config,
                    name=name,
                    batch_size=kwargs.pop("batch_size", default_batch_size),
                    transforms=kwargs.pop("transforms", default_transforms),
                    **kwargs,
                )
                for name, kwargs in entries.items()
            ],
        )
