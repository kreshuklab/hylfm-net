from typing import Any, Dict, List

from hylfm.metrics import Metric, get_metric
from hylfm.metrics.base import MetricValue


def init_metrics(metrics_config: List[Dict[str, Any]]) -> List[Metric]:
    return [get_metric(**kwargs) for kwargs in metrics_config]


def compute_metrics_individually(metrics: List[Metric], tensors: Dict) -> Dict[str, MetricValue]:
    out = {}
    for m in metrics:
        m.update(tensors)
        computed = m.compute()
        m.reset()

        assert isinstance(computed, dict)
        for k, v in computed.items():
            assert k not in out, (k, list(out.keys()))
            out[k] = v

    return out
