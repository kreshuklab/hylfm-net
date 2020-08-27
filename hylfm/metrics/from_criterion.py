from typing import Type

from hylfm.metrics.base import Metric


class MetricFromCriterion(Metric):
    def __init__(
        self,
        *super_args,
        loss_class: Type[GenericLossOnTensors],
        loss_kwargs: Dict[str, Any],
        tensor_names: Optional[Union[List[str], Dict[str, str]]],
        **super_kwargs,
    ):
        """note: override 'update_impl if custom_tensor_names need to be specified"""
        metric_tensor_names = {"meta": "meta"}
        if isinstance(tensor_names, dict):
            metric_tensor_names.update({expected_name: expected_name for expected_name in tensor_names.values()})
        else:
            metric_tensor_names.update({expected_name: expected_name for expected_name in tensor_names})

        super().__init__(*super_args, tensor_names=metric_tensor_names, **super_kwargs)
        self.loss = loss_class(tensor_names=tensor_names, **loss_kwargs)

    def reset(self):
        self._accumulated = 0.0
        self._n = 0

    def update_impl(self, **tensors) -> None:
        b = len(tensors["meta"])
        assert b > 0
        self._n += b
        with torch.no_grad():
            self._accumulated += float(self.loss(tensors).item())

    def compute_impl(self):
        loss_name = self.loss.name
        return {loss_name: self._accumulated / self._n}
