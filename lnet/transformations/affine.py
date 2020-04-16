import warnings
from typing import Any, List, Optional, OrderedDict, Sequence, Tuple, Union

import numpy
import torch.nn.functional
import torch.nn.functional
from scipy.ndimage import affine_transform

from lnet.utils.affine import inv_scipy_form2torch_form_2d, inv_scipy_form2torch_form_3d


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

        matrix = numpy.zeros((4, 4))
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

    def __init__(
        self,
        *,
        apply_to: str,
        target_to_compare_to: str,
        order: int,
        input_shape: Sequence[int],
        bdv_affine_transformations: List[List[float]],  # Fiij's big data viewer affine transformations: each affine transformation as a list of 12 floats.
        # affine_matrices: List[List[float]],
        output_shape: Sequence[int],
        inverted: bool = False,
        crop_out: Optional[Tuple[Tuple[int, int], ...]] = None,
        crop_in: Optional[Tuple[Tuple[int, int], ...]] = None,
    ):
        if len(input_shape) not in (2, 3):
            raise NotImplementedError

        if len(output_shape) not in (2, 3):
            raise NotImplementedError

        super().__init__()
        self.apply_to = apply_to
        self.target_to_compare_to = target_to_compare_to
        self.input_shape = input_shape
        self.output_shape = output_shape

        self.mode = self.mode_from_order[order]

        trf_matrices = [bdv_trafo_to_affine_matrix(m) for m in bdv_affine_transformations]
        trf_matrix = trf_matrices[0]
        for m in trf_matrices[1:]:
            trf_matrix = trf_matrix.dot(m)

        self.affine_grids = {}

        self.forward = self.inverted if inverted else self._forward
        if crop_out is None:
            self.order = 0
            ones_out = self._impl(
                numpy.ones((1, 1) + self.input_shape, dtype=numpy.uint8),
                matrix=numpy.linalg.inv(trf_matrix),
                trf_in_shape=self.input_shape,
                trf_out_shape=self.output_shape,
            )[0, 0]
            dims = []
            if len(ones_out.shape) == 3:
                dims.append(ones_out.max(1).max(1))
                ones_out_2d = ones_out.max(0)
                dims.append(ones_out_2d.max(1))
                dims.append(ones_out_2d.max(0))
            else:
                raise NotImplementedError

            crop_out = [(numpy.argmax(dim), -numpy.argmax(dim[::-1])) for dim in dims]
            print("determined crop_out:", crop_out)
        elif len(crop_out) == len(output_shape) + 1:
            assert crop_out[0][0] == 0 and crop_out[0][1] == 0, crop_out
            crop_out = crop_out[1:]

        if crop_in is None:
            crop_in = tuple([(0, 0) for _ in range(len(input_shape))])
        elif len(crop_in) == len(input_shape) + 1:
            assert crop_in[0][0] == 0 and crop_in[0][1] == 0, crop_in
            crop_in = crop_in[1:]

        self.z_offset = crop_out[0][0]
        self.cropped_output_shape = tuple(
            [c[1] - c[0] if c[1] > 0 else outs - c[0] + c[1] for outs, c in zip(self.output_shape, crop_out)]
        )
        self.cropped_input_shape = tuple(
            [c[1] - c[0] if c[1] > 0 else ins - c[0] + c[1] for ins, c in zip(self.input_shape, crop_in)]
        )

        self.order = order
        crop_shift_in = numpy.eye(len(input_shape) + 1, dtype=trf_matrix.dtype)
        crop_shift_in[:-1, -1] = [-c[0] for c in crop_in]
        crop_shift_out = numpy.eye(len(output_shape) + 1, dtype=trf_matrix.dtype)
        crop_shift_out[:-1, -1] = [c[0] for c in crop_out]
        self.trf_matrix = crop_shift_in.dot(trf_matrix.dot(crop_shift_out))

    #         self.inv_trf_matrix = numpy.linalg.inv(self.trf_matrix)

    @staticmethod
    def get_affine_grid(scipy_form, ipt_shape, out_shape):
        assert len(scipy_form.shape) == 2, scipy_form.shape
        assert len(ipt_shape) in (2, 3)
        assert len(ipt_shape) == len(out_shape), (ipt_shape, out_shape)
        theta = scipy_form2torch_theta(scipy_form, ipt_shape, out_shape)
        affine_grid_size = (1, 1) + tuple(out_shape)
        return torch.nn.functional.affine_grid(theta=theta, size=affine_grid_size, align_corners=False)

    def _impl(
        self,
        *,
        ipt: Union[torch.Tensor, numpy.ndarray],
        matrix: numpy.ndarray,
        trf_in_shape: Tuple[int, ...],
        trf_out_shape: Tuple[int, ...],
        output_sampling_shape: Optional[Tuple[int, ...]] = None,
        z_slices: Optional[Sequence[int]] = None,
    ) -> Union[numpy.ndarray, torch.Tensor]:
        print("ipt shape", ipt.shape)
        print("trf in shape", trf_in_shape)
        print("trf out shape", trf_out_shape)
        print("out sampling shape", output_sampling_shape)
        if output_sampling_shape is None:
            output_sampling_shape = trf_out_shape
        elif z_slices is not None and any([zs is not None for zs in z_slices]):
            raise ValueError("exclusive args: z_slices, output_sampling_shape")

        if trf_in_shape != ipt.shape[2:]:
            in_scaling = [ipts / trf_in for ipts, trf_in in zip(ipt.shape[2:], trf_in_shape)] + [1.0]
            print("ipt.shape -> trf_in_shape", in_scaling)
            matrix = numpy.diag(in_scaling).dot(matrix)

        if trf_out_shape != output_sampling_shape:
            out_scaling = [trf_out / outs for trf_out, outs in zip(trf_out_shape, output_sampling_shape)] + [1.0]
            print("trf_out_shape -> output_sampling", out_scaling)
            matrix = matrix.dot(numpy.diag(out_scaling))

        if isinstance(ipt, numpy.ndarray):
            assert len(ipt.shape) in [4, 5], ipt.shape
            return numpy.stack(
                [
                    numpy.stack(
                        [
                            affine_transform(ipt_woc, matrix, output_shape=output_sampling_shape, order=self.order)
                            for ipt_woc in ipt_wc
                        ]
                    )
                    for ipt_wc in ipt
                ]
            )
        elif isinstance(ipt, torch.Tensor):
            on_cuda = False
            ipt_was_cuda = ipt.is_cuda
            if on_cuda != ipt.is_cuda:
                if on_cuda:
                    ipt = ipt.to(torch.device("cuda:0"))
                else:
                    ipt = ipt.to(torch.device("cpu"))

            affine_grid_key = (matrix.tostring(), ipt.shape[2:], output_sampling_shape)
            affine_grid = self.affine_grids.get(affine_grid_key, None)
            if affine_grid is None:
                affine_grid = self.get_affine_grid(matrix, ipt.shape[2:], output_sampling_shape)
                affine_grid.to(ipt)
                self.affine_grids[affine_grid_key] = affine_grid

            affine_grid = affine_grid.to(ipt)

            if z_slices is None or all([zs is None for zs in z_slices]):
                affine_grid = affine_grid.expand(ipt.shape[0], *([-1] * (len(ipt.shape) - 1)))
            else:
                assert all([zs is not None for zs in z_slices]), z_slices
                assert len(z_slices) == ipt.shape[0], (z_slices, ipt.shape)
                assert all(self.z_offset <= z_slice for z_slice in z_slices), (self.z_offset, z_slices)
                affine_grid = torch.cat(
                    [
                        self.affine_torch_grid[:, z_slice - self.z_offset : z_slice + 1 - self.z_offset]
                        for z_slice in z_slices
                    ]
                )

            ret = torch.nn.functional.grid_sample(
                ipt, affine_grid, align_corners=False, mode=self.mode, padding_mode="zeros"  # "border"
            )
            if not (z_slices is None or all([zs is None for zs in z_slices])):
                assert ret.shape[2] == 1, ret.shape
                ret = ret[:, :, 0]

            if on_cuda == ipt_was_cuda:
                return ret
            elif ipt_was_cuda:
                return ret.to(device=torch.device("cuda:0"))
            else:
                return ret.to(device=torch.device("cpu"))
        else:
            raise TypeError(type(ipt))

    def inverted(self, ipt: Union[torch.Tensor, numpy.ndarray], **kwargs) -> OrderedDict[str, Any]:
        raise NotImplementedError
        ipt = tensors[self.apply_to]
        tgt = tensors[self.target_to_compare_to]

        return self._impl(
            ipt, matrix=self.trf_matrix, trf_in_shape=self.output_shape, trf_out_shape=self.input_shape, **kwargs
        )

    def _forward(self, img_shape, tensors: OrderedDict[str, Any]) -> OrderedDict[str, Any]:
        #         for meta in tensors["meta"]:
        #             tmeta = meta[self.target_to_compare_to]
        #             assert tmeta["shape_before_resize"][1:] == self.cropped_output_shape, (
        #                 tmeta["shape_before_resize"],
        #                 self.cropped_output_shape,
        #             )

        #         output_sampling_shape = (
        #             None  # sample output as trf output and select z_slice (for 2d target) ...
        #             if len(tensors[self.target_to_compare_to].shape) == 4
        #             else tuple(tensors[self.target_to_compare_to].shape[2:])  # ...or resample output to compare to volumetric target
        #         )
        tensors[self.apply_to] = self._impl(
            ipt=tensors[self.apply_to],
            matrix=self.trf_matrix,
            trf_in_shape=self.cropped_input_shape,
            trf_out_shape=self.cropped_output_shape,
            #             output_sampling_shape=(84, 133, 162),
            #             output_sampling_shape=(100, 200),
            output_sampling_shape=(100, 200, 300),
            z_slices=[m.get("z_slice", None) for m in tensors["meta"]],
        )
        return tensors


if __name__ == "__main__":
    # trf = AffineTransformation(apply_to="lr", target_to_compare_to="ls", order=0, input_shape=(838, 1330, 1615), output_shape=(241, 1501, 1801), matrices=[[0.98048,0.004709,0.098297,-111.7542,7.6415e-05,0.97546,0.0030523,-20.1143,0.014629,8.2964e-06,-3.9928,846.8515]], crop_in=[(5, -5), (10, -10), (10, -10)], crop_out=[(3, -22), (20, -117), (87, -41)])
    trf = AffineTransformation(
        apply_to="lr",
        target_to_compare_to="ls",
        order=0,
        input_shape=(838, 1330, 1615),
        output_shape=(241, 1501, 1801),
        bdv_affine_transformations=[
            [
                0.98048,
                0.004709,
                0.098297,
                -111.7542,
                7.6415e-05,
                0.97546,
                0.0030523,
                -20.1143,
                0.014629,
                8.2964e-06,
                -3.9928,
                846.8515,
            ]
        ],
        crop_in=((0, 0), (0, 0), (0, 0)),
        crop_out=((0, 0), (0, 0), (0, 0)),
    )
