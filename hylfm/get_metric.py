import copy
import logging
from functools import partial
from inspect import getfullargspec
from typing import Any, Dict, Tuple, Union

import hylfm.plain_metrics
import hylfm.plain_criteria


logger = logging.getLogger(__name__)


def get_metric_class_and_kwargs(name, kwargs: Dict[str, Any]) -> Tuple[hylfm.plain_metrics.base.Metric, Dict[str, Any]]:
    metric_class = getattr(hylfm.plain_metrics, name, None)
    if metric_class is None:
        metric_class = hylfm.plain_metrics.LossAsMetric
        assert "loss_kwargs" in kwargs
        assert "loss_class" not in kwargs
        kwargs = {"loss_class": getattr(hylfm.plain_criteria, name), **kwargs}

    return metric_class, kwargs


def init_metric(name: str, kwargs: Dict[str, Any]):
    kwargs = copy.copy(kwargs)
    kwargs["postfix"] = kwargs.get("postfix", "").format_map(kwargs)
    along_dim = kwargs.pop("along_dim", None)
    dim_len = kwargs.pop("dim_len", None)

    if along_dim is None:
        metric = metric_class(**metric_kwargs)
    else:
        tensor_names = metric_kwargs.pop("tensor_names", {})
        metric = hylfm.plain_metrics.AlongDimMetric(
            along_dim=along_dim, dim_len=dim_len, tensor_names=tensor_names, metric_class=metric_class, **metric_kwargs
        )

    return metric


def get_metric(name: str, kwargs: Dict[str, Any]):
    try:
        metric = init_metric(name, kwargs)
    except TypeError as e:
        logger.error("Could not init %s with %s", name, kwargs)
        raise e

    return metric
