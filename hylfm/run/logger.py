from __future__ import annotations

import collections
import logging
from enum import Enum
from typing import Any, Callable, Dict, List

import torch
import numpy
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
    convert_metrics: Callable[[Dict[str, Any]], Dict[str, Any]]

    def __init__(self, *, log_every: Period = Period(1, PeriodUnit.iteration)):
        self.log_every = log_every
        self.last_batch_len = 0

    def __call__(self, *, epoch: int, epoch_len: int, iteration: int, batch_len: int, **metrics) -> None:
        if self.log_every.match(epoch=epoch, iteration=iteration, epoch_len=epoch_len):
            self.log_metrics(epoch=epoch, epoch_len=epoch_len, iteration=iteration, batch_len=batch_len, **metrics)

        self.last_batch_len = batch_len

    def log_metrics(self, *, epoch: int, epoch_len: int, iteration: int, batch_len: int, **metrics):
        raise NotImplementedError

    def log_summary(self, **metrics):
        raise NotImplementedError


class WandbEvalLogger(RunLogger):
    def __init__(self, point_cloud_threshold: float = 0.2, **super_kwargs):
        super().__init__(**super_kwargs)
        self.point_cloud_threshold = point_cloud_threshold
        self.tables = collections.defaultdict(list)
        self.hist = collections.defaultdict(list)

    def log_metrics_sample(self, *, epoch: int, epoch_len: int, iteration: int, batch_idx: int, **metrics):
        assert epoch == 0
        conv = {}
        for key, value in metrics.items():
            if isinstance(value, list):
                raise TypeError(key)

            elif "idx_pos" in key:
                raise NotImplementedError("idx_pos")
                assert isinstance(value, numpy.ndarray)
                assert len(value.shape) == 2
                assert value.shape[1] == 3

            elif isinstance(value, (torch.Tensor, numpy.ndarray)):
                if isinstance(value, torch.Tensor):
                    value = value.cpu().numpy()

                if len(value.shape) == 1:
                    self.tables[(key[-1], key[:-2])] += [[d, v] for d, v in enumerate(value)]
                elif len(value.shape) == 4:
                    c, z, y, x = value.shape
                    assert z > 1
                    assert y > 1
                    assert x > 1
                    if c == 1:
                        value = numpy.concatenate([value] * 3)

                    if c == 2:
                        mask = value.max(0) > self.point_cloud_threshold
                        idx = numpy.asarray(numpy.where(mask)).T
                        pixels = value[numpy.broadcast_to(mask[None], value.shape)].reshape(-1, 2)
                        color = numpy.ones(len(pixels))
                        color[pixels[:, 0] > self.point_cloud_threshold] -= 1
                        color[pixels[:, 1] > self.point_cloud_threshold] += 1

                        assert len(idx) == len(color)
                        point_cloud = numpy.asarray([list(coord) + [col] for coord, col in zip(idx, color)])
                    elif c == 3:
                        mask = value.sum(0) > self.point_cloud_threshold
                        idx = numpy.asarray(numpy.where(mask)).T
                        rgb = value[numpy.broadcast_to(mask[None], value.shape)].reshape(-1, 3)
                        assert len(idx) == len(rgb)
                        point_cloud = numpy.asarray([list(coord) + [r, g, b] for coord, (r, g, b) in zip(idx, rgb)])
                    else:
                        raise NotImplementedError(c)

                    conv[key] = wandb.Object3D(point_cloud)

            elif isinstance(value, (float, int)):
                conv[key] = value
                self.hist[key].append(value)
            else:
                raise NotImplementedError((key, value))

        step = (epoch * epoch_len + iteration) * self.last_batch_len + batch_idx
        wandb.log(conv, step=step)

    def log_metrics(self, *, epoch: int, epoch_len: int, iteration: int, batch_len: int, **metrics):
        for key, val in metrics.items():
            assert isinstance(val, list), key
            assert len(val) == batch_len, (key, len(val))

        for i in range(batch_len):
            self.log_metrics_sample(
                epoch=epoch,
                epoch_len=epoch_len,
                iteration=iteration,
                batch_idx=i,
                **{k: v[i] for k, v in metrics.items()}
            )

    def log_summary(self, **metrics):
        summary = {}
        final_log = {}
        for key, value in metrics.items():
            if isinstance(value, list):
                raise TypeError(key)

            elif "idx_pos" in key:
                raise NotImplementedError("idx_pos")
                assert isinstance(value, numpy.ndarray)
                assert len(value.shape) == 2
                assert value.shape[1] == 3

            elif isinstance(value, numpy.ndarray):
                assert len(value.shape) == 1
                dim = key[-1]
                name = key[:-2] + "_avg"
                table = wandb.Table(data=[[d, v] for d, v in enumerate(value)], columns=[dim, name])
                final_log[key + "_scatter_avg"] = wandb.plot.scatter(table=table, x=dim, y=name)

            elif isinstance(value, (float, int)):
                summary[key] = value
            else:
                raise NotImplementedError((key, value))

        for (dim, key), data in self.tables.items():
            table = wandb.Table(data=data, columns=[dim, key])
            final_log[key + "_scatter"] = wandb.plot.scatter(table=table, x=dim, y=key)

        for key, data in self.hist.items():
            data = [[s] for s in data]
            table = wandb.Table(data=data, columns=[key])
            final_log[key + "_hist"] = wandb.plot.histogram(table, key, title="hist: " + key)

        wandb.log(final_log)
        wandb.summary.update(summary)


class MultiLogger(RunLogger):
    def __init__(self, loggers: List[RunLogger], **super_kwargs):
        super().__init__(**super_kwargs)
        self.loggers = [globals().get(lgr)(**super_kwargs) for lgr in loggers]

    def log_metrics(self, *, step: int, **metrics):
        for lgr in self.loggers:
            lgr.log_metrics(step=step, **metrics)
