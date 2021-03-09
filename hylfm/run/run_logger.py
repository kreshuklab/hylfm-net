from __future__ import annotations

import collections
import logging
from typing import Any, Callable, Dict, Optional, Tuple

import matplotlib.cm
import numpy
import pandas
import PIL.Image
import torch
import wandb

from hylfm.hylfm_types import MetricChoice
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

    def __init__(
        self,
        *,
        log_every: Period = Period(1, PeriodUnit.iteration),
        step: Optional[int] = None,  # set to overwrite step given in log methods
    ):
        self.log_every = log_every
        self.step = step

    def __call__(self, *, epoch: int, iteration: int, epoch_len: int, step: int, **metrics) -> None:
        if self.step is not None:
            step = self.step

        if self.log_every.match(epoch=epoch, iteration=iteration, epoch_len=epoch_len):
            self.log_metrics(step=step, **metrics)

    def log_metrics(self, *, step: int, **metrics):
        raise NotImplementedError

    def log_summary(self, *, step: int, **metrics):
        raise NotImplementedError


class WandbLogger(RunLogger):
    def __init__(self, *, point_cloud_threshold: float, zyx_scaling: Tuple[float, float, float], **super_kwargs):
        super().__init__(**super_kwargs)
        self.point_cloud_threshold = point_cloud_threshold
        self.tables = collections.defaultdict(list)
        self.hist = collections.defaultdict(list)
        self.zyx_scaling = zyx_scaling

    @torch.no_grad()
    def log_metrics_sample(self, *, step: int, **metrics):
        conv = {}

        def log_img(img):
            img = img.transpose(1, 2, 0)
            y, x, c = img.shape
            if c == 1:
                img = matplotlib.cm.cividis(img[..., 0])
            elif c == 2:
                alpha = img.max(-1).clip(0, 1)
                balance = (img[..., 0].clip(0, 1) - img[..., 1].clip(0, 1)) / 2 + 0.5
                img = numpy.concatenate([matplotlib.cm.viridis(balance)[..., :3], alpha[..., None]], -1)
            elif c > 4:
                raise NotImplementedError(c)

            img = (img * 255).clip(0, 255).astype(numpy.uint8)
            pil_img = PIL.Image.fromarray(img)
            conv[key] = wandb.Image(pil_img, caption=key)

        def log_point_cloud(img):
            c, z, y, x = img.shape
            assert z > 1
            assert y > 1
            assert x > 1
            if c == 1:
                img = numpy.concatenate([img] * 3)
                c = 3

            if c == 2:
                mask = img.max(0) > self.point_cloud_threshold
                if not mask.max():
                    logger.debug("no points in cloud")
                    return

                idx = numpy.asarray(numpy.where(mask)).T
                idx = idx * numpy.broadcast_to(numpy.array([self.zyx_scaling]), idx.shape)
                pixels = img[numpy.broadcast_to(mask[None], img.shape)].reshape(-1, 2)
                color = numpy.ones(len(pixels))
                color[pixels[:, 0] > self.point_cloud_threshold] += 1
                color[pixels[:, 1] > self.point_cloud_threshold] += 2

                assert len(idx) == len(color)
                point_cloud = numpy.asarray([list(coord) + [col] for coord, col in zip(idx, color)])
                conv[key] = wandb.Object3D(point_cloud)
            elif c == 3:
                raise NotImplementedError("result looks only black and white!")
                mask = img.sum(0) > self.point_cloud_threshold
                if not mask.max():
                    logger.debug("no points in cloud")
                    return

                idx = numpy.asarray(numpy.where(mask)).T
                idx = idx * numpy.broadcast_to(numpy.array([self.zyx_scaling]), idx.shape)
                rgb = img[numpy.broadcast_to(mask[None], img.shape)].reshape(-1, 3)
                assert len(idx) == len(rgb)
                point_cloud = numpy.asarray([list(coord) + [r, g, b] for coord, (r, g, b) in zip(idx, rgb)])
                conv[key] = wandb.Object3D(point_cloud)
            else:
                raise NotImplementedError(c)

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
                    value = value.detach().cpu().numpy()

                if len(value.shape) == 1:
                    self.tables[(key[-1], key[:-2])] += [[d, v] for d, v in enumerate(value)]
                elif len(value.shape) == 4:
                    c, z, y, x = value.shape
                    if z == 1:
                        log_img(value[:, 0])
                    else:
                        log_point_cloud(value)

                elif len(value.shape) == 3:
                    log_img(value)
                else:
                    raise NotImplementedError(value.shape)
            elif isinstance(value, (float, int)):
                conv[key] = value
                self.hist[key].append(value)
            else:
                raise NotImplementedError((key, value))

        wandb.log(conv, step=step)

    def log_metrics(self, *, step: int, **metrics):
        if self.step is not None:
            step = self.step

        sample_metrics = {k: v for k, v in metrics.items() if isinstance(v, list)}
        batch_metrics = {k: v for k, v in metrics.items() if not isinstance(v, list)}

        self.log_metrics_sample(step=step, **{k: v for k, v in batch_metrics.items()})

        if sample_metrics:
            for i in range(min([len(v) for v in sample_metrics.values()])):
                self.log_metrics_sample(step=step + i, **{k: v[i] for k, v in sample_metrics.items()})

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
                summary[key] = value.mean().item()

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
            summary[key] = numpy.asarray(data).mean().item()

        return final_log, summary

    def log_summary(self, *, step: int, **metrics):
        if self.step is not None:
            step = self.step

        final_log, summary = self._get_final_log_and_summary(metrics)
        wandb.log(final_log, step=step)
        wandb.summary.update(summary)


class WandbValidationLogger(WandbLogger):
    def __init__(self, *, score_metric: MetricChoice, minimize: bool, step: int = 0, **super_kwargs):
        super().__init__(
            **super_kwargs, step=step
        )  # no optional step, overwrite with training step to log validation at correct step
        self.score_metric = score_metric
        self.minimize = minimize
        self.best_score = None
        self.val_it = 0

    def log_metrics(self, *, step: int, **metrics):
        # don't log metrics per step when validating
        pass

    def log_summary(self, *, step: int, **metrics):
        if self.step is not None:
            step = self.step

        self.val_it += 1
        final_log, summary = self._get_final_log_and_summary(metrics)
        final_log["it"] = self.val_it

        final_log.update(summary)
        metrics = {"val_" + k: v for k, v in final_log.items()}
        wandb.log(metrics, step=step)

        score = summary[self.score_metric.value]
        if self.minimize:
            score *= -1

        if self.best_score is None or self.best_score < score:
            self.best_score = score
            summary["it"] = self.val_it
            wandb.summary.update({"val_" + k: v for k, v in summary.items()})
