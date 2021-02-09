import logging
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import torch
from torch.utils.data import DataLoader

from hylfm import __version__
from hylfm.hylfm_types import TransformLike
from hylfm.metrics.base import MetricGroup
from hylfm.run.run_logger import RunLogger


class Run:
    name: str
    log_path: Optional[Path]

    def __init__(
        self,
        *,
        batch_postprocessing: TransformLike,
        batch_premetric_trf: TransformLike,
        batch_preprocessing_in_step: TransformLike,
        dataloader: torch.utils.data.DataLoader,
        batch_size: int,
        metrics: MetricGroup,
        model: Optional[torch.nn.Module],
        name: Optional[str] = None,
        run_logger: RunLogger,
        tgt_name: Optional[str],
    ):
        assert isinstance(__version__, str)
        self.dirty = "dirty" in __version__
        if self.dirty:
            warnings.warn(f"uncommited changes in version {__version__}")

        logging.getLogger(__name__).info("run version: %s", __version__)

        if model is not None and torch.cuda.is_available():
            model = model.cuda(0)

        self.model = model
        self.dataloader = dataloader
        self.batch_size = batch_size
        self.batch_preprocessing_in_step = batch_preprocessing_in_step
        self.batch_postprocessing = batch_postprocessing
        self.batch_premetric_trf = batch_premetric_trf

        self.metrics = metrics
        self.tgt_name = tgt_name

        self.run_logger = run_logger

        self.name = self.__class__.__name__ if name is None else name

    def __iter__(self):
        for batch in self._run():
            yield batch

    def run(self):
        for it in self:
            pass

    def _run(self) -> Iterable[Dict[str, Any]]:
        raise NotImplementedError
