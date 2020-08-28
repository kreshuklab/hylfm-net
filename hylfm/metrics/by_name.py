import logging
from typing import Any, Dict, Optional, Tuple

import hylfm.losses
from hylfm.metrics import Metric
from hylfm.metrics.along_dim import AlongDimMetric
from hylfm.metrics.from_loss import MetricFromLoss

logger = logging.getLogger(__name__)


def get_metric(
    name: str, *args, postfix: str = "", along_dim__dim_len: Optional[Tuple[int, int]] = None, **kwargs: Dict[str, Any]
) -> Metric:
    kwargs = {"postfix": postfix.format_map({"along_dim__dim_len": along_dim__dim_len, **kwargs}), **kwargs}
    if hasattr(hylfm.metrics, name):
        metric_class = getattr(hylfm.metrics, name)
    elif hasattr(hylfm.losses, name):
        metric_class = MetricFromLoss
        assert "loss_class" not in kwargs
        kwargs["loss_class"] = getattr(hylfm.losses, name)
    else:
        raise NotImplementedError(name)

    if along_dim__dim_len is not None:
        kwargs["metric_class"] = metric_class
        kwargs["along_dim__dim_len"] = along_dim__dim_len
        metric_class = AlongDimMetric

    try:
        return metric_class(*args, **kwargs)
    except TypeError as e:
        logger.error("Could not init %s (%s) with %s", metric_class, name, kwargs)
        raise e
