import typing
from typing import Any

import lnet.plain_metrics
import lnet.plain_criteria


def get_metric(name: str, kwargs: typing.Dict[str, Any]):
    postfix_idx = name.find("-")
    if postfix_idx == -1:
        postfix = ""
    elif postfix_idx == 0:
        raise NotImplementedError("negativ metric value?")
    else:
        postfix = name[postfix_idx:]
        name = name[:postfix_idx]

    metric_class = getattr(lnet.plain_metrics, name, None)
    if metric_class is None:
        loss_class = getattr(lnet.plain_criteria, name)
        metric = lnet.plain_metrics.LossAsMetric(loss_class(**kwargs), postfix=postfix)
    else:
        metric = metric_class(postfix=postfix, **kwargs)

    return metric
