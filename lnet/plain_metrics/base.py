import typing
from collections import OrderedDict
from copy import deepcopy

import numpy
import torch.nn

from lnet.plain_criteria import GenericLossOnTensors
from lnet.utils.general import camel_to_snake


class Metric:
    def __init__(
        self,
        *,
        postfix: str = "",
        tensor_names: typing.Optional[typing.Dict[str, str]],
        scale: typing.Optional[str] = None,
        to_minimize: typing.Optional[str] = None,  # for scale
        vs: typing.Optional[str] = None,  # for scale
    ):
        if scale is not None:
            assert to_minimize is not None
            assert vs is not None

            self.vs_norm = f"{vs}_normalized"
            scaled = f"scaled_{scale}_to_minimize_{to_minimize}_vs_{self.vs_norm}"
            if isinstance(tensor_names, list):
                tensor_names = [
                    scaled if expected_name == scale else self.vs_norm if expected_name == vs else expected_name
                    for expected_name in tensor_names
                ]
            else:
                tensor_names = {
                    name: scaled if expected_name == scale else self.vs_norm if expected_name == vs else expected_name
                    for name, expected_name in tensor_names.items()
                }

            scale_fn_name = f"scale_to_minimize_{to_minimize}"
            if not hasattr(self, scale_fn_name):
                raise NotImplementedError(scale_fn_name)

            self.scale = getattr(self, scale_fn_name)
            self.map_unscaled = {scale: scaled}
            self.map_unnormed = {vs: self.vs_norm}
            postfix += "-scaled"
        else:
            self.scale = None
            self.map_unscaled = {}
            self.map_unnormed = {}
            self.vs_norm = None

        self.to_minimize = to_minimize

        self.tensor_names = tensor_names

        self.postfix = postfix
        self.reset()

    def reset(self):
        raise NotImplementedError

    def prepare_for_update(self, tensors: typing.OrderedDict) -> typing.OrderedDict:
        for unnormed, normed in self.map_unnormed.items():
            if normed not in tensors:
                tensors[normed] = self.norm(tensors[unnormed])

        for unscaled, scaled in self.map_unscaled.items():
            if scaled not in tensors:
                tensors[scaled] = self.scale(tensors[unscaled], tensors[self.vs_norm])

        if isinstance(self.tensor_names, list):
            assert all([expected_name in tensors for expected_name in self.tensor_names]), (
                self.tensor_names,
                list(tensors.keys()),
            )
        else:
            assert all([expected_name in tensors for expected_name in self.tensor_names.values()]), (
                self.tensor_names,
                list(tensors.keys()),
            )

        return tensors

    def update(self, tensors: typing.OrderedDict) -> None:
        tensors = self.prepare_for_update(tensors)

        # def clone_tensor(tensor):
        #     if isinstance(tensor, torch.Tensor):
        #         return tensor.clone()
        #     elif isinstance(tensor, numpy.ndarray):
        #         return numpy.copy(tensor)
        #     elif isinstance(tensor, list):
        #         return deepcopy(tensor)
        #     else:
        #         raise NotImplementedError(type(tensor))

        self.update_impl(
            **{name: tensors[expected_name] for name, expected_name in self.tensor_names.items()}
        )  # todo: remove clone and test that metric does not change input data instead

    def update_impl(self, **kwargs) -> None:
        raise NotImplementedError

    def compute(self) -> typing.Dict[str, float]:
        computed = self.compute_impl()
        if isinstance(computed, float):
            return {camel_to_snake(self.__class__.__name__) + self.postfix: float(computed)}
        elif isinstance(computed, (dict, OrderedDict)):
            return {key + self.postfix: value for key, value in computed.items()}
        else:
            raise NotImplementedError(type(computed))

    def compute_impl(self) -> typing.Union[float, typing.Dict[str, float]]:
        raise NotImplementedError

    @staticmethod
    def scale_to_minimize_mse(
        ipt: typing.Union[numpy.ndarray, torch.Tensor], vs_norm: typing.Union[numpy.ndarray, torch.Tensor]
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

    @staticmethod
    def scale_to_minimize_mae(
        ipt: typing.Union[numpy.ndarray, torch.Tensor], vs_norm: typing.Union[numpy.ndarray, torch.Tensor]
    ):
        raise NotImplementedError
        # ipt_numpy = ipt if isinstance(ipt, numpy.ndarray) else ipt.numpy()
        # vs_norm_numpy = vs_norm if isinstance(vs_norm, numpy.ndarray) else vs_norm.numpy()
        #
        # ipt_numpy = ipt_numpy.astype(numpy.float32, copy=False)
        # vs_norm_numpy = vs_norm_numpy.astype(numpy.float32, copy=False)
        #
        # ipt_mean = ipt_numpy.mean()
        # scale = numpy.cov(ipt_numpy.flatten() - ipt_mean, vs_norm_numpy.flatten())[0, 1] / numpy.var(
        #     ipt_numpy.flatten()
        # )
        # return (ipt - ipt_mean) * scale

    @staticmethod
    def norm(unnormed: typing.Union[numpy.ndarray, torch.Tensor]):
        return unnormed - unnormed.mean()


# class ScaledMetric(Metric):  # todo: add as optional fields in Metric
#     def __init__(
#         self,
#         *super_args,
#         tensor_names: typing.Optional[typing.Dict[str, str]],
#         scale: str,
#         to_minimize: str,
#         vs: str,
#         **super_kwargs,
#     ):
#         # assert scale in tensor_names.values()
#         # assert vs in tensor_names.values()
#         scaled = f"scaled_{scale}_to_minimize_{to_minimize}_vs_{vs}"
#         scaled_tensor_names = {
#             name: scaled if expected_name == scale else expected_name for name, expected_name in tensor_names.items()
#         }
#         super().__init__(*super_args, tensor_names=scaled_tensor_names, **super_kwargs)
#         self.map_unscaled = {scale: scaled}
#         scale_fn_name = f"scale_to_minimize_{to_minimize}"
#         if not hasattr(self, scale_fn_name):
#             raise NotImplementedError(scale_fn_name)
#
#         self.scale = getattr(self, scale_fn_name)
#         self.vs = vs
#
#     def update(self, tensors: typing.OrderedDict) -> None:
#         for unscaled, scaled in self.map_unscaled.items():
#             if scaled not in tensors:
#                 tensors[scaled] = self.scale(tensors[unscaled], tensors[self.vs])
#
#         super().update(tensors)


class LossAsMetric(Metric):
    def __init__(
        self,
        *super_args,
        loss_class: typing.Type[GenericLossOnTensors],
        loss_kwargs: typing.Dict[str, typing.Any],
        tensor_names: typing.Optional[typing.Union[typing.List[str], typing.Dict[str, str]]],
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


# class ScaledLossAsMetric(ScaledMetric, LossAsMetric):
#     pass


class AlongDimMetric(Metric):
    def __init__(
        self,
        *,
        along_dim: int,
        dim_len: int,
        metric_class,
        dim_names: typing.Optional[typing.Dict[int, str]] = None,
        postfix: str = "",
        tensor_names: typing.Optional[typing.Dict[str, str]],
        scale: typing.Optional[str] = None,
        to_minimize: typing.Optional[str] = None,  # for scale
        vs: typing.Optional[str] = None,  # for scale,
        **sub_metric_kwargs,
    ):

        self.submetrics = []  # for reset
        super().__init__(postfix=postfix, tensor_names=tensor_names, scale=scale, to_minimize=to_minimize, vs=vs)
        self.submetrics = [metric_class(tensor_names=self.tensor_names, **sub_metric_kwargs) for i in range(dim_len)]
        self.along_dim = along_dim
        assert dim_len > 0

        if dim_names is None:
            dim_names = {0: "batch", 1: "channel", 2: "z", 3: "y", 4: "x"}

        self.postfix += f"-along_{dim_names[along_dim]}"

    def reset(self):
        for metric in self.submetrics:
            metric.reset()

    def update(self, tensors: typing.OrderedDict) -> None:
        tensors = self.prepare_for_update(tensors)
        subtensors = list(self.slice_along(self.along_dim, tensors))
        assert len(subtensors) == len(self.submetrics), (len(subtensors), len(self.submetrics))
        for metric, stensors in zip(self.submetrics, subtensors):
            metric.update(stensors)

    def compute(self) -> typing.Dict[str, typing.List[float]]:
        subresults = [metric.compute() for metric in self.submetrics]
        subresults_keys = {k for k in subresults[0]}
        result = {k + self.postfix: [] for k in subresults_keys}
        for sr in subresults:
            assert set(sr.keys()) == subresults_keys
            for k, v in sr.items():
                result[k + self.postfix].append(v)

        return result

    @staticmethod
    def slice_along(slice_dim: int, tensors: dict):
        batch_len = None
        slice_dim_len = None

        for name, tensor in tensors.items():
            if isinstance(tensor, list):
                if batch_len is None:
                    batch_len = len(tensor)
                else:
                    assert batch_len == len(tensor)
            else:
                if batch_len is None:
                    batch_len = tensor.shape[0]
                else:
                    assert batch_len == tensor.shape[0]

                assert len(tensor.shape) > slice_dim
                if slice_dim_len is None:
                    slice_dim_len = tensor.shape[slice_dim]
                else:
                    assert slice_dim_len == tensor.shape[slice_dim], (slice_dim_len, tensor.shape)

        slice_meta_key = f"slice_at_dim_{slice_dim}"
        for i in range(slice_dim_len):
            slice_tensors = {
                k: v if isinstance(v, list) else v[tuple([slice(None)] * slice_dim + [i])] for k, v in tensors.items()
            }
            for meta in slice_tensors["meta"]:
                assert slice_meta_key not in meta, meta

            slice_tensors["meta"] = [{slice_meta_key: i, **meta} for meta in slice_tensors["meta"]]

            yield slice_tensors
