import logging
from typing import Any, Collection, Dict, List, Optional, Sequence, Tuple, Union

import numpy
import torch.nn.functional
import torch.nn.functional
from scipy.ndimage import affine_transform

from hylfm.transforms.affine_utils import (
    MAX_SRHINK_IN_LENSLETS,
    get_bdv_affine_transformations_by_name,
    get_lf_shape,
    get_ls_ref_shape,
    get_ls_roi,
    get_raw_lf_shape,
)

logger = logging.getLogger(__name__)


def scipy_form2torch_theta(scipy_form, ipt_shape, out_shape) -> torch.Tensor:
    assert len(scipy_form.shape) == 2
    assert not scipy_form[-1, :-1].any()
    assert scipy_form[-1, -1] == 1
    norm_i = numpy.diag([2 / s for s in ipt_shape] + [1])
    norm_i[:-1, -1] = -1
    norm_o = numpy.diag(list(out_shape) + [2])
    norm_o[:-1, -1] = out_shape
    norm_o = norm_o / 2
    scipy_normed = norm_i.dot(scipy_form).dot(norm_o)
    assert not scipy_normed[-1, :-1].any()

    # transpose axis to match scipy implementation
    theta4x4 = numpy.zeros_like(scipy_normed)
    theta4x4[:-1, :-1] = scipy_normed[-2::-1, -2::-1]
    theta4x4[:-1, -1] = scipy_normed[-2::-1, -1]
    theta4x4[-1, -1] = scipy_normed[-1, -1]

    # return with batch dimension as 1x3x4
    return torch.from_numpy(theta4x4[None, :-1])


def bdv_trafo_to_affine_matrix(trafo):
    """from https://github.com/constantinpape/elf/blob/7b7cd21e632a07876a1302dad92f8d7c1929b37a/elf/transformation/affine.py#L162
    Translate bdv transformation (XYZ) to affine matrix (ZYX)

    """
    if len(trafo) == 12:
        assert trafo[10] != 0.0
        assert trafo[5] != 0.0
        assert trafo[0] != 0.0
        sub_matrix = numpy.zeros((3, 3))
        sub_matrix[0, 0] = trafo[10]
        sub_matrix[0, 1] = trafo[9]
        sub_matrix[0, 2] = trafo[8]

        sub_matrix[1, 0] = trafo[6]
        sub_matrix[1, 1] = trafo[5]
        sub_matrix[1, 2] = trafo[4]

        sub_matrix[2, 0] = trafo[2]
        sub_matrix[2, 1] = trafo[1]
        sub_matrix[2, 2] = trafo[0]

        shift = [trafo[11], trafo[7], trafo[3]]

        matrix = numpy.zeros((4, 4), dtype="float64")
        matrix[:3, :3] = sub_matrix
        matrix[:3, 3] = shift
        matrix[3, 3] = 1

        return matrix
    elif len(trafo) == 6:
        raise NotImplementedError("just a guess...")
        assert trafo[4] != 0.0
        assert trafo[0] != 0.0
        matrix = numpy.eye(3)
        matrix[0, 0] = trafo[4]
        matrix[0, 1] = trafo[3]
        matrix[1, 0] = trafo[1]
        matrix[1, 1] = trafo[0]
        matrix[0, 2] = trafo[5]
        matrix[1, 2] = trafo[2]
        return matrix
    else:
        raise NotImplementedError(trafo)


class AffineTransformation(torch.nn.Module):
    mode_from_order = {0: "nearest", 2: "bilinear"}
    scipy_padding_mode = {"zeros": "constant", "border": "nearest"}

    def __init__(
        self,
        *,
        apply_to: Union[str, Dict[str, str]],
        target_to_compare_to: Union[
            str, Tuple[int, str], Tuple[Union[int, float], Union[int, float], Union[int, float]]
        ],
        order: int,
        align_corners: bool,
        # subtract_one_when_scale: str,
        ref_input_shape: Union[str, Sequence[int]],
        bdv_affine_transformations: Union[
            str, List[List[float]]
        ],  # Fiij's big data viewer affine transformations: each affine transformation as a list of 12 floats.
        # affine_matrices: List[List[float]],
        ref_output_shape: Union[str, Sequence[int]],
        ref_crop_in: Optional[Tuple[Tuple[int, Optional[int]], ...]] = None,
        ref_crop_out: Optional[Tuple[Tuple[int, Optional[int]], ...]] = None,
        inverted: bool = False,
        padding_mode: str = "border",
    ):
        self.align_corners = align_corners
        # self.subtract_one_when_scale = subtract_one_when_scale
        if isinstance(ref_input_shape, str):
            ref_input_shape = [838] + get_lf_shape(ref_input_shape)
        if len(ref_input_shape) not in (2, 3):
            raise NotImplementedError

        if isinstance(ref_output_shape, str):
            ref_output_shape = get_ls_ref_shape(ref_output_shape)
        if len(ref_output_shape) not in (2, 3):
            raise NotImplementedError

        super().__init__()
        if isinstance(apply_to, str):
            apply_to = {apply_to: apply_to}

        self.apply_to: Dict[str, str] = apply_to
        if not isinstance(target_to_compare_to, str) and all(
            [not isinstance(ttct, str) for ttct in target_to_compare_to]
        ):
            target_to_compare_to_float = tuple(target_to_compare_to)
            target_to_compare_to = tuple([int(t) for t in target_to_compare_to])
            assert all(
                [tf == ti for tf, ti in zip(target_to_compare_to_float, target_to_compare_to)]
            ), target_to_compare_to_float

        self.target_to_compare_to = target_to_compare_to
        self.input_shape = tuple(ref_input_shape)
        self.output_shape = tuple(ref_output_shape)

        self.mode = self.mode_from_order[order]

        if isinstance(bdv_affine_transformations, str):
            bdv_affine_transformations = get_bdv_affine_transformations_by_name(bdv_affine_transformations)

        assert len(bdv_affine_transformations) >= 1
        trf_matrices = [bdv_trafo_to_affine_matrix(m) for m in bdv_affine_transformations]
        trf_matrix = trf_matrices[0]
        for m in trf_matrices[1:]:
            trf_matrix = trf_matrix.dot(m)

        self.affine_grids = {}

        self.forward = self._inverted if inverted else self._forward

        if ref_crop_in is None:
            ref_crop_in = tuple([(0, None) for _ in range(len(ref_input_shape))])
        else:
            assert len(ref_crop_in) == len(ref_input_shape), (ref_crop_in, ref_input_shape)
        # elif len(ref_crop_in) == len(ref_input_shape) + 1:
        #     assert ref_crop_in[0][0] == 0 and ref_crop_in[0][1] == 0, ref_crop_in
        #     ref_crop_in = ref_crop_in[1:]

        self.cropped_input_shape = tuple(
            [
                ins - c[0] if c[1] is None else c[1] - c[0] if c[1] > 0 else ins - c[0] + c[1]
                for ins, c in zip(self.input_shape, ref_crop_in)
            ]
        )

        crop_shift_in = numpy.eye(len(ref_input_shape) + 1, dtype=trf_matrix.dtype)
        crop_shift_in[:-1, -1] = [-c[0] for c in ref_crop_in]
        trf_matrix = crop_shift_in.dot(trf_matrix)

        if ref_crop_out is None:
            self.order = 0
            self.padding_mode = "zeros"
            ones_out = self._impl(
                ipt=numpy.ones((1, 1) + self.input_shape, dtype=numpy.uint8),
                matrix=trf_matrix,
                trf_in_shape=tuple(self.cropped_input_shape),
                trf_out_shape=tuple(self.output_shape),
                order=0,
            )[0, 0]
            ones_out = ones_out.astype(bool)
            dims = []
            roi_method = "max"
            if roi_method == "tight":
                raise NotImplementedError
            elif roi_method == "max":
                if len(ones_out.shape) == 3:
                    dims.append(ones_out.max(1).max(1))
                    ones_out_2d = ones_out.max(0)
                    dims.append(ones_out_2d.max(1))
                    dims.append(ones_out_2d.max(0))
                else:
                    raise NotImplementedError
                ref_crop_out = [(numpy.argmax(dim), -numpy.argmax(dim[::-1])) for dim in dims]
            else:
                raise NotImplementedError(roi_method)

            logger.warning("determined crop_out: %s", ref_crop_out)
        elif len(ref_crop_out) == len(ref_output_shape) + 1:
            assert ref_crop_out[0][0] == 0 and ref_crop_out[0][1] is None or ref_crop_out[0][1] == 0, ref_crop_out
            ref_crop_out = ref_crop_out[1:]

        self.padding_mode = padding_mode
        self.z_offset = ref_crop_out[0][0]
        self.cropped_output_shape = tuple(
            [
                outs - c[0] if c[1] is None else c[1] - c[0] if c[1] > 0 else outs - c[0] + c[1]
                for outs, c in zip(self.output_shape, ref_crop_out)
            ]
        )

        crop_shift_out = numpy.eye(len(ref_output_shape) + 1, dtype=trf_matrix.dtype)
        crop_shift_out[:-1, -1] = [c[0] for c in ref_crop_out]
        self.trf_matrix = trf_matrix.dot(crop_shift_out)
        self.trf_matrix_inv = numpy.linalg.inv(self.trf_matrix)
        self.order = order

    @staticmethod
    def get_affine_grid(scipy_form, ipt_shape, out_shape, align_corners):
        assert len(scipy_form.shape) == 2, scipy_form.shape
        assert len(ipt_shape) in (2, 3)
        assert len(ipt_shape) == len(out_shape), (ipt_shape, out_shape)
        theta = scipy_form2torch_theta(scipy_form, ipt_shape, out_shape)
        affine_grid_size = (1, 1) + tuple(out_shape)
        return torch.nn.functional.affine_grid(theta=theta, size=affine_grid_size, align_corners=align_corners)

    def _impl(
        self,
        *,
        ipt: Union[torch.Tensor, numpy.ndarray],
        matrix: numpy.ndarray,
        trf_in_shape: Tuple[int, ...],
        trf_out_shape: Tuple[int, ...],
        order: int,
        output_sampling_shape: Optional[Tuple[int, ...]] = None,
        z_slices: Optional[Sequence[int]] = None,
    ) -> Union[numpy.ndarray, torch.Tensor]:
        logger.debug("ipt shape %s", ipt.shape)
        logger.debug("trf in shape %s", trf_in_shape)
        logger.debug("trf out shape %s", trf_out_shape)
        logger.debug("out sampling shape %s", output_sampling_shape)
        if output_sampling_shape is None:
            output_sampling_shape = trf_out_shape
        elif z_slices is not None and any([zs is not None for zs in z_slices]):
            assert output_sampling_shape[0] > max(z_slices) - self.z_offset, (
                output_sampling_shape,
                z_slices,
                self.z_offset,
            )

        if trf_in_shape != ipt.shape[2:]:
            in_scaling = [ipts / trf_in for ipts, trf_in in zip(ipt.shape[2:], trf_in_shape)] + [1.0]
            logger.debug("ipt.shape -> trf_in_shape %s %s %s", ipt.shape[2:], trf_in_shape, in_scaling)
            matrix = numpy.diag(in_scaling).dot(matrix)

        if trf_out_shape != output_sampling_shape:
            out_scaling = [trf_out / outs for trf_out, outs in zip(trf_out_shape, output_sampling_shape)] + [1.0]
            logger.debug("trf_out_shape -> output_sampling %s %s %s", trf_out_shape, output_sampling_shape, out_scaling)
            matrix = matrix.dot(numpy.diag(out_scaling))

        matrix = matrix.astype("float32")
        if isinstance(ipt, numpy.ndarray):
            assert len(ipt.shape) in [4, 5], ipt.shape
            ret = numpy.stack(
                [
                    numpy.stack(
                        [
                            affine_transform(
                                ipt_woc,
                                matrix,
                                output_shape=output_sampling_shape,
                                order=order,
                                mode=self.scipy_padding_mode[self.padding_mode],
                            )
                            for ipt_woc in ipt_wc
                        ]
                    )
                    for ipt_wc in ipt
                ]
            )
            if z_slices is None or all([zs is None for zs in z_slices]):
                return ret
            else:
                return numpy.stack([b[:, zs - self.z_offset : zs + 1 - self.z_offset] for b, zs in zip(ret, z_slices)])
        elif isinstance(ipt, torch.Tensor):
            on_cuda = ipt.is_cuda
            ipt_was_cuda = ipt.is_cuda
            if on_cuda != ipt.is_cuda:
                if on_cuda:
                    ipt = ipt.to(torch.device("cuda:0"))
                else:
                    ipt = ipt.to(torch.device("cpu"))

            affine_grid_key = (matrix.tostring(), ipt.shape[2:], output_sampling_shape)
            affine_grid = self.affine_grids.get(affine_grid_key, None)
            if affine_grid is None:
                affine_grid = self.get_affine_grid(matrix, ipt.shape[2:], output_sampling_shape, self.align_corners)
                affine_grid.to(ipt)
                self.affine_grids[affine_grid_key] = affine_grid

            affine_grid = affine_grid.to(ipt)

            if z_slices is None or all([zs is None for zs in z_slices]):
                affine_grid = affine_grid.expand(ipt.shape[0], *([-1] * (len(ipt.shape) - 1)))
            else:
                assert all([zs is not None for zs in z_slices]), z_slices
                assert len(z_slices) == ipt.shape[0], (z_slices, ipt.shape)
                assert all(self.z_offset <= z_slice for z_slice in z_slices), (self.z_offset, z_slices)
                assert affine_grid.shape[0] == 1
                affine_grid = torch.cat(
                    [affine_grid[:, z_slice - self.z_offset : z_slice + 1 - self.z_offset] for z_slice in z_slices]
                )

            ret = torch.nn.functional.grid_sample(
                ipt, affine_grid, align_corners=self.align_corners, mode=self.mode, padding_mode=self.padding_mode
            )

            if on_cuda == ipt_was_cuda:
                return ret
            elif ipt_was_cuda:
                return ret.to(device=torch.device("cuda:0"))
            else:
                return ret.to(device=torch.device("cpu"))
        else:
            raise TypeError(type(ipt))

    def _inverted(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(self.target_to_compare_to, str):
            z_slices = batch["z_slice"]
            output_sampling_shape = batch[self.target_to_compare_to].shape[2:]
        elif any([isinstance(ttct, str) for ttct in self.target_to_compare_to]):
            z_slices = None
            ttct_list_of_lists = [
                batch[ttct].shape[2 + i :] if isinstance(ttct, str) else [ttct]
                for i, ttct in enumerate(self.target_to_compare_to)
            ]
            output_sampling_shape = tuple([ttct for ttct_list in ttct_list_of_lists for ttct in ttct_list])
        else:
            z_slices = None
            output_sampling_shape = tuple(self.target_to_compare_to)

        assert 0 not in output_sampling_shape, output_sampling_shape
        assert len(output_sampling_shape) == 3, output_sampling_shape

        if output_sampling_shape[0] == 1:
            # single z_slice:
            output_sampling_shape = (self.cropped_input_shape[0],) + output_sampling_shape[1:]

        for in_name, out_name in self.apply_to.items():
            batch[out_name] = self._impl(
                ipt=batch[in_name],
                matrix=self.trf_matrix_inv,
                trf_in_shape=self.cropped_output_shape,
                trf_out_shape=self.cropped_input_shape,
                order=self.order,
                output_sampling_shape=output_sampling_shape,
                z_slices=z_slices,
            )
        return batch

    def _forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(self.target_to_compare_to, str):
            z_slices = batch["z_slice"]
            output_sampling_shape = batch[self.target_to_compare_to].shape[2:]
        elif any([isinstance(ttct, str) for ttct in self.target_to_compare_to]):
            z_slices = None
            ttct_list_of_lists = [
                batch[ttct].shape[2 + i :] if isinstance(ttct, str) else [ttct]
                for i, ttct in enumerate(self.target_to_compare_to)
            ]
            output_sampling_shape = tuple([ttct for ttct_list in ttct_list_of_lists for ttct in ttct_list])
        else:
            z_slices = None
            output_sampling_shape = tuple(self.target_to_compare_to)

        assert 0 not in output_sampling_shape, output_sampling_shape
        assert len(output_sampling_shape) == 3, output_sampling_shape

        if output_sampling_shape[0] == 1:
            # single z_slice:
            output_sampling_shape = (self.cropped_output_shape[0],) + output_sampling_shape[1:]

        for in_name, out_name in self.apply_to.items():
            batch[out_name] = self._impl(
                ipt=batch[in_name],
                matrix=self.trf_matrix,
                trf_in_shape=self.cropped_input_shape,
                trf_out_shape=self.cropped_output_shape,
                order=self.order,
                output_sampling_shape=output_sampling_shape,
                z_slices=z_slices,
            )
        return batch


class AffineTransformationDynamicTraining(torch.nn.Module):
    def __init__(
        self,
        apply_to: str,  # todo: only allow 'str' for apply_to
        *,
        target_to_compare_to: str,
        crop_names: Collection[str],
        pred_z_min: int,
        pred_z_max: int,
        nnum: int,
        z_ls_rescaled: int,
        scale: int,
        padding_mode: str = "border",
        interpolation_order: int = 2,
    ):
        super().__init__()
        self.apply_to = apply_to
        ops = {}
        for crop_name in crop_names:
            ref_input_shape = [838] + get_raw_lf_shape(crop_name, wrt_ref=True)
            ref_roi_in = [[pred_z_min, pred_z_max]] + [
                [MAX_SRHINK_IN_LENSLETS * nnum, s - MAX_SRHINK_IN_LENSLETS * nnum] for s in ref_input_shape[1:]
            ]
            ref_roi_out = get_ls_roi(
                crop_name, for_slice=False, nnum=nnum, wrt_ref=True, z_ls_rescaled=z_ls_rescaled, ls_scale=scale
            )

            ops[crop_name] = AffineTransformation(
                apply_to=apply_to,
                target_to_compare_to=target_to_compare_to,
                order=interpolation_order,
                ref_input_shape=ref_input_shape,
                bdv_affine_transformations=crop_name,
                ref_output_shape=get_ls_ref_shape(crop_name),
                ref_crop_in=ref_roi_in,  # noqa
                ref_crop_out=ref_roi_out,  # noqa
                inverted=False,
                padding_mode=padding_mode,
                align_corners=False,
            )

        self.ops = torch.nn.ModuleDict(ops)

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        return self.ops[batch["crop_name"]](batch)
