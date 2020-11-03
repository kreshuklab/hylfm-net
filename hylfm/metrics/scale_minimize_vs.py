from typing import Dict, Optional, Tuple, Union

import numpy
import torch

from hylfm.metrics import Metric


class ScaleMinimizeVsMetric(Metric):
    def __init__(
        self, *super_args, tensor_names, scale_minimize_vs: Optional[Tuple[str, str, str]] = None, **super_kwargs
    ):
        if scale_minimize_vs is not None:
            scale, minimize, vs = scale_minimize_vs
            self.vs_norm = f"{vs}_normalized"
            scaled = f"scaled_{scale}_to_minimize_{minimize}_vs_{self.vs_norm}"
            tensor_names = {
                name: scaled if expected_name == scale else self.vs_norm if expected_name == vs else expected_name
                for name, expected_name in tensor_names.items()
            }

            scale_fn_name = f"scale_minimize_{minimize}"
            if not hasattr(self, scale_fn_name):
                raise NotImplementedError(scale_fn_name)

            self.scale = getattr(self, scale_fn_name)
            self.map_unnormed = {vs: self.vs_norm}
            self.map_unscaled = {scale: scaled}
            add_to_postfix = "-scaled"
        else:
            self.scale = None
            self.map_unnormed = {}
            self.map_unscaled = {}
            self.vs_norm = None
            add_to_postfix = ""

        super().__init__(*super_args, tensor_names=tensor_names, **super_kwargs)
        self.postfix += add_to_postfix

    def prepare_for_update(self, tensors: Dict) -> Dict:
        for unnormed, normed in self.map_unnormed.items():
            if normed not in tensors:
                tensors[normed] = self.norm(tensors[unnormed])

        for unscaled, scaled in self.map_unscaled.items():
            if scaled not in tensors:
                tensors[scaled] = self.scale(tensors[unscaled], tensors[self.vs_norm])

        return super().prepare_for_update(tensors)

    @classmethod
    def scale_minimize_mse(cls, ipt: Union[numpy.ndarray, torch.Tensor], vs_norm: Union[numpy.ndarray, torch.Tensor]):
        scaled = [
            cls._scale_minimize_mse_sample(ipt_sample, vs_norm_sample)
            for ipt_sample, vs_norm_sample in zip(ipt, vs_norm)
        ]
        if isinstance(ipt, numpy.ndarray):
            stack = numpy.stack
        elif isinstance(ipt, torch.Tensor):
            stack = torch.stack
        else:
            raise TypeError(type(ipt))

        return stack(scaled)

    @staticmethod
    def _scale_minimize_mse_sample(
        ipt: Union[numpy.ndarray, torch.Tensor], vs_norm: Union[numpy.ndarray, torch.Tensor]
    ):
        ipt_numpy = ipt if isinstance(ipt, numpy.ndarray) else ipt.numpy()
        vs_norm_numpy = vs_norm if isinstance(vs_norm, numpy.ndarray) else vs_norm.numpy()

        ipt_numpy = ipt_numpy.astype(numpy.float32, copy=False)
        vs_norm_numpy = vs_norm_numpy.astype(numpy.float32, copy=False)

        ipt_mean = ipt_numpy.mean()
        scale = numpy.cov(ipt_numpy.flatten() - ipt_mean, vs_norm_numpy.flatten())[0, 1] / numpy.var(
            ipt_numpy.flatten()
        )
        return (ipt - ipt_mean) * scale
