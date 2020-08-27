from typing import Optional, Tuple, Union

import numpy
import torch


class ScaleMinimizeVsMixin:
    postfix: str

    def __init__(
        self, *super_args, tensor_names, scale_minimize_vs: Optional[Tuple[str, str, str]] = None, **super_kwargs
    ):
        if scale_minimize_vs is not None:
            scale, minimize, vs = scale_minimize_vs
            self.vs_norm = f"{vs}_normalized"
            scaled = f"scaled_{scale}_to_minimize_{minimize}_vs_{self.vs_norm}"
            self.tensor_names = {
                name: scaled if expected_name == scale else self.vs_norm if expected_name == vs else expected_name
                for name, expected_name in tensor_names.items()
            }

            scale_fn_name = f"scale_minimize_{minimize}"
            if not hasattr(self, scale_fn_name):
                raise NotImplementedError(scale_fn_name)

            self.scale = getattr(self, scale_fn_name)
            self.map_unnormed = {vs: self.vs_norm}
            self.map_unscaled = {scale: scaled}
            self.postfix += "-scaled"
        else:
            self.tensor_names = tensor_names
            self.scale = None
            self.map_unnormed = {}
            self.map_unscaled = {}
            self.vs_norm = None

        super().__init__(*super_args, **super_kwargs)

    @staticmethod
    def scale_minimize_mse(ipt: Union[numpy.ndarray, torch.Tensor], vs_norm: Union[numpy.ndarray, torch.Tensor]):
        ipt_numpy = ipt if isinstance(ipt, numpy.ndarray) else ipt.numpy()
        vs_norm_numpy = vs_norm if isinstance(vs_norm, numpy.ndarray) else vs_norm.numpy()

        ipt_numpy = ipt_numpy.astype(numpy.float32, copy=False)
        vs_norm_numpy = vs_norm_numpy.astype(numpy.float32, copy=False)

        ipt_mean = ipt_numpy.mean()
        scale = numpy.cov(ipt_numpy.flatten() - ipt_mean, vs_norm_numpy.flatten())[0, 1] / numpy.var(
            ipt_numpy.flatten()
        )
        return (ipt - ipt_mean) * scale
