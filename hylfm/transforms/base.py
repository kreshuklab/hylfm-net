import logging
from typing import Any, Dict, List, Tuple, Union
from hylfm.datasets.collate import collate, separate
from hylfm.hylfm_types import TransformLike

try:
    from typing import Protocol
except ImportError:
    from typing_extensions import Protocol

logger = logging.getLogger(__name__)


class Transform:
    randomly_changes_shape: bool = False

    def __init__(
        self,
        apply_to: Union[str, List[str], Tuple[str], Dict[str, str]] = None,
        input_mapping: Dict[str, str] = None,
        output_mapping: Dict[str, str] = None,
    ):
        if apply_to is None:
            self.input_mapping = input_mapping or {}
            self.output_mapping = output_mapping or {}
        else:
            assert input_mapping is None
            assert output_mapping is None
            if isinstance(apply_to, (list, tuple)):
                self.input_mapping = dict(zip(apply_to, apply_to))
                self.output_mapping = {}
            elif isinstance(apply_to, str):
                self.input_mapping = {apply_to: "tensor"}
                self.output_mapping = {"tensor": apply_to}
            elif isinstance(apply_to, dict) and len(apply_to) == 1:
                k, v = next(iter(apply_to.items()))
                self.input_mapping = {k: "tensor"}
                self.output_mapping = {"tensor": v}
            else:
                raise TypeError(type(apply_to), apply_to)

    def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        missing_inputs = [k for k in self.input_mapping if k not in batch]
        if missing_inputs:
            raise ValueError(f"required batch keys {missing_inputs} not found in batch {list(batch.keys())}")

        input_batch = {self.input_mapping[k]: v for k, v in batch.items() if k in self.input_mapping}
        try:
            transformed = self.apply_to_batch(**input_batch)
        except Exception:
            logger.error("transform %s failed", self)
            raise

        if isinstance(transformed, dict):
            for k, v in transformed.items():
                batch[self.output_mapping.get(k, k)] = v
        elif len(self.output_mapping) == 1:
            k, v = next(iter(self.output_mapping.items()))
            batch[v] = transformed
        else:
            raise NotImplementedError(f"{self}, {type(transformed)}")

        return batch

    def apply_to_batch(self, **batch: Any) -> Dict[str, Any]:
        transformed_samples = [self.apply_to_sample(**sample_in) for sample_in in separate(batch)]

        # output mapping done in self.__call__(), but we need valid samples of type dict for collate()
        for i, ts in enumerate(transformed_samples):
            if isinstance(ts, dict):
                pass
            elif len(self.output_mapping) == 1:
                k = next(iter(self.output_mapping))
                transformed_samples[i] = {k: ts}
            else:
                raise NotImplementedError(type(ts))

        return collate(transformed_samples)

    def apply_to_sample(self, **sample: Any) -> Any:
        raise RuntimeError(f"{self}.apply_to_sample() not implemented or called erroneously")

    def __add__(self, other):
        if isinstance(other, ComposedTransform):
            return ComposedTransform(self, *other.transforms)
        elif isinstance(other, Transform):
            return ComposedTransform(self, other)
        else:
            raise NotImplementedError(type(other))


class ComposedTransform(Transform):
    def __init__(self, *transforms: TransformLike):
        assert all([callable(transform) for transform in transforms])
        super().__init__()
        self.transforms = list(transforms)
        self.update_randomly_changes_shape()

    def update_randomly_changes_shape(self):
        self.randomly_changes_shape = any(getattr(t, "randomly_changes_shape", True) for t in self.transforms)

    def __add__(self, other):
        if isinstance(other, self.__class__):
            return self.__class__(*self.transforms, *other.transforms)
        elif isinstance(other, Transform):
            return self.__class__(*self.transforms, other)
        else:
            raise NotImplementedError(type(other))

    def __iadd__(self, other):
        if isinstance(other, self.__class__):
            self.transforms += other.transforms
        elif isinstance(other, Transform):
            self.transforms.append(other)
        else:
            raise NotImplementedError(type(other))

        self.update_randomly_changes_shape()
        return self

    def remove(self, name):
        transform_idx = None
        for idx, transform in enumerate(self.transforms):
            if type(transform).__name__ == name:
                transform_idx = idx
                break
        if transform_idx is not None:
            self.transforms.pop(transform_idx)

        self.update_randomly_changes_shape()
        return self

    def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        for transform in self.transforms:
            batch = transform(batch)
            assert isinstance(batch, dict), transform

        return batch


class DTypeMapping:
    DTYPE_MAPPING = {
        "float32": "float32",
        "float": "float32",
        "double": "float64",
        "float64": "float64",
        "half": "float16",
        "float16": "float16",
        "long": "int64",
        "int64": "int64",
        "byte": "uint8",
        "uint8": "uint8",
        "int": "int32",
        "int32": "int32",
    }
