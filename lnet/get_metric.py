import copy
import logging
import typing
from functools import partial
from typing import Any

import lnet.plain_metrics
import lnet.plain_criteria


logger = logging.getLogger(__name__)


def get_metric(name: str, kwargs: typing.Dict[str, Any]):
    kwargs = copy.copy(kwargs)
    postfix_idx = name.find("-")
    if postfix_idx == -1:
        postfix = ""
    elif postfix_idx == 0:
        raise NotImplementedError("negativ metric value?")
    else:
        postfix = name[postfix_idx:]
        name = name[:postfix_idx]

    along_dim = kwargs.pop("along_dim", None)
    dim_len = kwargs.pop("dim_len", None)
    try:
        metric_class = getattr(lnet.plain_metrics, name, None)
        if metric_class is None:
            loss_class = getattr(lnet.plain_criteria, name)
            metric_kwargs = {
                kw: kwargs.pop(kw) for kw in ["tensor_names", "scale", "to_minimize", "vs"] if kw in kwargs
            }
            metric_class = partial(lnet.plain_metrics.LossAsMetric, loss_class=loss_class, loss_kwargs=kwargs)
            # (postfix=postfix, **metric_kwargs, loss_class=loss_class, loss_kwargs=kwargs)
        else:
            metric_kwargs = kwargs

        metric_kwargs["postfix"] = postfix
        if along_dim is None:
            metric = metric_class(**metric_kwargs)
        else:
            tensor_names = metric_kwargs.pop("tensor_names", {})
            metric = lnet.plain_metrics.AlongDimMetric(
                along_dim=along_dim,
                dim_len=dim_len,
                tensor_names=tensor_names,
                metric_class=metric_class,
                **metric_kwargs,
            )
    # if loss_class is None:
    #     loss_class = getattr(lnet.plain_criteria, name.replace("Scaled", ""))
    # assert scaled_kwargs["scale"] == kwargs.get("pred", "pred"), (scaled_kwargs, kwargs)
    # LossAsMetricClass = partial(lnet.plain_metrics.ScaledLossAsMetric, **scaled_kwargs)
    # assert "prefix" not in kwargs
    # kwargs["prefix"] = "Scaled"
    # tensor_names = kwargs.get("tensor_names", {})
    # tensor_names[
    #     "pred"
    # ] = f"scaled_{tensor_names.get('pred', 'pred')}_to_minimize_{scaled_kwargs['to_minimize']}_vs_{scaled_kwargs['vs']}"
    # kwargs["tensor_names"] = tensor_names
    # else:
    #     LossAsMetricClass = lnet.plain_metrics.LossAsMetric
    except TypeError as e:
        logger.error("Could not init %s with %s", name, kwargs)
        raise e

    return metric
