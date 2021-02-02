import logging
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import torch
from torch.utils.data import DataLoader

from hylfm import __version__
from hylfm.hylfm_types import TransformLike
from hylfm.metrics.base import MetricGroup
from hylfm.run.logger import RunLogger


class Run:
    name: str
    log_path: Optional[Path]

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        batch_preprocessing_in_step: TransformLike,
        batch_postprocessing: TransformLike,
        batch_premetric_trf: TransformLike,
        metrics: MetricGroup,
        pred_name: str,
        tgt_name: Optional[str],
        run_logger: RunLogger,
        name: Optional[str] = None,
    ):
        assert isinstance(__version__, str)
        self.dirty = "dirty" in __version__
        if self.dirty:
            warnings.warn(f"uncommited changes in version {__version__}")

        logging.getLogger(__name__).info("run version: %s", __version__)

        if torch.cuda.is_available():
            model = model.cuda(0)

        self.model = model
        self.dataloader = dataloader
        self.batch_preprocessing_in_step = batch_preprocessing_in_step
        self.batch_postprocessing = batch_postprocessing
        self.batch_premetric_trf = batch_premetric_trf

        self.metrics = metrics
        self.pred_name = pred_name
        self.tgt_name = tgt_name

        self.run_logger = run_logger

        self.name = self.__class__.__name__ if name is None else name

    def __iter__(self):
        for batch in self._run():
            yield batch


    def _run(self) -> Iterable[Dict[str, Any]]:
        raise NotImplementedError
