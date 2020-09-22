import logging
from typing import Dict, List, Optional, Tuple, Type

from hylfm.metrics import Metric
from hylfm.metrics.scale_minimize_vs import ScaleMinimizeVsMetric

logger = logging.getLogger(__name__)


class AlongDimMetric(ScaleMinimizeVsMetric):
    def __init__(
        self,
        *,
        along_dim__dim_len: Tuple[int, int],
        metric_class: Type[Metric],
        postfix: str = "",
        tensor_names: Optional[Dict[str, str]],
        scale_minimize_vs: Optional[Tuple[str, str, str]] = None,
        scale_each: bool = False,
        **sub_metric_kwargs,
    ):
        if scale_each:
            sub_metric_kwargs["scale_minimize_vs"] = scale_minimize_vs
            scale_minimize_vs = None

        self.submetrics = []  # for reset call in super init
        super().__init__(postfix=postfix, tensor_names=tensor_names, scale_minimize_vs=scale_minimize_vs)

        along_dim, dim_len = along_dim__dim_len
        self.along_dim = along_dim
        assert dim_len > 0
        self.dim_len = dim_len
        self.submetrics = [metric_class(tensor_names=self.tensor_names, **sub_metric_kwargs) for _ in range(dim_len)]

        self.postfix += f"-along_dim_{along_dim}"

    def reset(self):
        for metric in self.submetrics:
            metric.reset()

    def update(self, tensors: Dict) -> None:
        assert isinstance(tensors, dict)
        tensors = self.prepare_for_update(tensors)
        assert isinstance(tensors, dict)
        for metric, sub_tensors in zip(self.submetrics, self.slice_along(self.along_dim, tensors)):
            metric.update(sub_tensors)

    def compute(self) -> Dict[str, List[float]]:
        subresults = [metric.compute() for metric in self.submetrics]
        subresult_keys = set(subresults[0].keys())
        result = {k + self.postfix: [] for k in subresult_keys}
        for sr in subresults:
            assert set(sr.keys()) == subresult_keys
            for k, v in sr.items():
                result[k + self.postfix].append(v)

        return result

    def slice_along(self, slice_dim: int, tensors: Dict):
        assert slice_dim >= 0
        slice_dim += 1  # add batch dim
        sliceable_tensors = set()
        for name, tensor in tensors.items():
            if not isinstance(tensor, list) and slice_dim < len(tensor.shape):
                if tensor.shape[slice_dim] == self.dim_len:
                    sliceable_tensors.add(name)
                else:
                    logger.debug(
                        "Not slicing along dim %s of tensor %s with b,shape %s", slice_dim - 1, name, tensor.shape
                    )

        assert sliceable_tensors
        slice_meta_key = f"slice_at_dim_{slice_dim - 1}"
        for i in range(self.dim_len):
            slice_tensors = {
                k: v if isinstance(v, list) or k not in sliceable_tensors else v[tuple([slice(None)] * slice_dim + [i])]
                for k, v in tensors.items()
            }
            if "meta" in slice_tensors:
                for meta in slice_tensors["meta"]:
                    assert slice_meta_key not in meta, meta

                slice_tensors["meta"] = [{slice_meta_key: i, **meta} for meta in slice_tensors["meta"]]

            yield slice_tensors
