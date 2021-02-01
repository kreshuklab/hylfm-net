from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Dict, List

import wandb

from hylfm.metrics.base import MetricGroup

logger = logging.getLogger(__name__)


class PeriodUnit(Enum):
    epoch = "epoch"
    iteration = "iteration"


class Period:
    def __init__(self, value: int, unit: PeriodUnit):
        self.value = value
        self.unit = unit

    def match(self, *, epoch: int, iteration: int, epoch_len: int):
        if self.unit == PeriodUnit.epoch:
            if epoch % self.value == 0 and iteration == epoch_len - 1:
                return True
        if self.unit == PeriodUnit.iteration:
            if iteration % self.value == 0:
                return True
        else:
            raise NotImplementedError(self.unit)

        return False


def log_exception(func):
    def wrap(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except Exception as e:
            logger.error(e, exc_info=True)

    return wrap


class RunLogger:
    def __init__(self, *, log_every: Period = Period(1, PeriodUnit.iteration), metrics: MetricGroup):
        self.log_every = log_every
        self.metrics = metrics

    def __call__(self, *, epoch: int, epoch_len: int, iteration: int, batch_len: int, **batch) -> Dict[str, Any]:
        if self.log_every.match(epoch=epoch, iteration=iteration, epoch_len=epoch_len):
            metrics = self.metrics.compute()
            step = epoch * epoch_len + iteration
            self.log_metrics(step=step, **metrics)
            batch.update(metrics)

        batch.update(epoch=epoch, epoch_len=epoch_len, iteration=iteration, batch_len=batch_len)
        return batch

    def log_metrics(self, *, step: int, **metrics):
        raise NotImplementedError


class WandbLogger(RunLogger):
    def log_metrics(self, *, step: int, **metrics):
        wandb.log(metrics, step=step)


class WandbSummaryLogger(RunLogger):
    def log_metrics(self, *, step: int, **metrics):
        wandb.summary.update(metrics)


class MultiLogger(RunLogger):
    def __init__(self, loggers: List[RunLogger], **super_kwargs):
        super().__init__(**super_kwargs)
        self.loggers = [globals().get(lgr)(**super_kwargs) for lgr in loggers]

    def log_metrics(self, *, step: int, **metrics):
        for lgr in self.loggers:
            lgr.log_metrics(step=step, **metrics)
