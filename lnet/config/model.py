from pathlib import Path
from typing import Type, Dict, Any, Optional

from attr import dataclass
from lnet import models
from lnet.models.lnet import LnetModel


@dataclass
class ModelConfig:
    Model: Type[LnetModel]
    kwargs: Dict[str, Any]
    nnum: int
    precision: str
    checkpoint: Optional[Path]

    name: str = None

    def __post_init__(self):
        assert self.precision == "float" or self.precision == "half"
        if self.name is None:
            self.name = self.Model.__name__

        if self.checkpoint is not None:
            assert self.checkpoint.exists(), self.checkpoint.absolute()
            assert self.checkpoint.is_file(), self.checkpoint.absolute()

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
