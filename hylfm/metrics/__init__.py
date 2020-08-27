import sys
from typing import Any, Dict, Optional, Tuple

import hylfm.criteria
from .base import Metric
from .beads import BeadPrecisionRecall
from .nrmse import NRMSE
from .psnr import PSNR
from .from_criterion import MetricFromCriterion


def _get_metric_class_and_kwargs(name, kwargs: Dict[str, Any]) -> Tuple[Metric, Dict[str, Any]]:
    metric_class = getattr(sys.modules[__name__], name, None)
    if metric_class is None:
        metric_class = MetricFromCriterion
        assert "criterion_class" not in kwargs
        kwargs = {"criterion_class": getattr(hylfm.criteria, name), **kwargs}

    return metric_class, kwargs


def get_metric(name: str, postfix: str = "", along_dim_dim_len: Optional[Tuple[int, int]] = None, **kwargs: Dict[str, Any]):
    kwargs["postfix"] = postfix.format_map(kwargs)
    metric_class, kwargs = _get_metric_class_and_kwargs(name, **kwargs)
    try:
        if along_dim_dim_len is None:
            metric = metric_class(**metric_kwargs)
        else:
            tensor_names = metric_kwargs.pop("tensor_names", {})
            metric = hylfm.plain_metrics.AlongDimMetric(
                along_dim=along_dim, dim_len=dim_len, tensor_names=tensor_names, metric_class=metric_class, **metric_kwargs
            )

        return metric

    except TypeError as e:
        logger.error("Could not init %s with %s", name, kwargs)
        raise e

    return metric