from __future__ import annotations

import collections
import logging
from typing import Any, Callable, Dict, List, Tuple

import torch
import numpy
import pandas
import wandb

from hylfm.utils.general import Period, PeriodUnit

logger = logging.getLogger(__name__)


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


class WandbLogger(RunLogger):
    def __init__(self, *, point_cloud_threshold: float = 0.2, zyx_scaling: Tuple[float, float, float], **super_kwargs):
        super().__init__(**super_kwargs)
        self.point_cloud_threshold = point_cloud_threshold
        self.tables = collections.defaultdict(list)
        self.hist = collections.defaultdict(list)
        self.zyx_scaling = zyx_scaling

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
                        idx = idx * numpy.broadcast_to(numpy.array([self.zyx_scaling]), idx.shape)
                        pixels = value[numpy.broadcast_to(mask[None], value.shape)].reshape(-1, 2)
                        color = numpy.ones(len(pixels))
                        color[pixels[:, 0] > self.point_cloud_threshold] -= 1
                        color[pixels[:, 1] > self.point_cloud_threshold] += 1

                        assert len(idx) == len(color)
                        point_cloud = numpy.asarray([list(coord) + [col] for coord, col in zip(idx, color)])
                    elif c == 3:
                        mask = value.sum(0) > self.point_cloud_threshold
                        idx = numpy.asarray(numpy.where(mask)).T
                        idx = idx * numpy.broadcast_to(numpy.array([self.zyx_scaling]), idx.shape)
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

    def _get_final_log_and_summary(self, metrics):
        summary = {}
        final_log = {}
        for key, value in metrics.items():
            if isinstance(value, list):
                raise TypeError(key)

            elif isinstance(value, pandas.DataFrame):
                final_log[key] = wandb.Table(dataframe=value)

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

        return final_log, summary

    def log_summary(self, **metrics):
        final_log, summary = self._get_final_log_and_summary(metrics)
        wandb.log(final_log)
        wandb.summary.update(summary)


class WandbValidationLogger(WandbLogger):
    def __init__(self, *, score_metric: str, minimize: bool, **super_kwargs):
        super().__init__(**super_kwargs)
        self.score_metric = score_metric
        self.minimize = minimize
        self.best_score = None
        self.val_it = 0

    def log_metrics(self, *, epoch: int, epoch_len: int, iteration: int, batch_len: int, **metrics):
        # don't log metrics per step when validating
        pass

    def log_summary(self, **metrics):
        self.val_it += 1
        final_log, summary = self._get_final_log_and_summary(metrics)
        final_log["it"] = self.val_it
        wandb.log({"val_" + k: v for k, v in final_log.items()}, commit=False)

        score = summary[self.score_metric]
        if self.minimize:
            score *= -1

        if self.best_score is None or self.best_score < score:
            self.best_score = score
            summary["it"] = self.val_it
            wandb.summary.update({"val_" + k: v for k, v in summary.items()})


class MultiLogger(RunLogger):
    def __init__(self, loggers: List[RunLogger], **super_kwargs):
        super().__init__(**super_kwargs)
        self.loggers = [globals().get(lgr)(**super_kwargs) for lgr in loggers]

    def log_metrics(self, *, step: int, **metrics):
        for lgr in self.loggers:
            lgr.log_metrics(step=step, **metrics)
