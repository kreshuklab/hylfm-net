from typing import Any, List, Optional, OrderedDict, Sequence, Tuple, Union

import numpy
import torch.nn.functional
import torch.nn.functional
from scipy.ndimage import affine_transform

from lnet.utils.affine import inv_scipy_form2torch_form_2d, inv_scipy_form2torch_form_3d


class AffineTransformation(torch.nn.Module):
    mode_from_order = {0: "nearest", 2: "bilinear"}

    def __init__(
        self,
        *,
        apply_to: str,
        target_to_compare_to: str,
        order: int,
        input_shape: Sequence[int],
        matrices: List[List[float]],
        output_shape: Sequence[int],
        inverted: bool = False,
        crop: Optional[Tuple[int, ...]] = None,
    ):
        super().__init__()
        self.input_2_transform = apply_to
        self.target_to_compare_to = target_to_compare_to

        self.input_shape = input_shape
        self.output_shape = output_shape

        self.mode = self.mode_from_order[order]

        trf_matrix = self.concat_affine_matrices([self.bdv_trafo_to_affine_matrix(at) for at in matrices])
        inv_trf_matrix = numpy.linalg.inv(trf_matrix)
        # self._forward_matrix = numpy.linalg.inv(matrix_with__forward_crop)
        self.affine_grid_size = None
        # self.z_offset: int = self._forward_crop[0][0]

        self.forward = self.inverted if inverted else self._forward
        if crop is None:
            self.order = 0
            ones_out = self._impl(
                numpy.ones((1, 1) + input_shape, dtype=numpy.uint8),
                matrix=inv_trf_matrix,
                trf_in_shape=input_shape,
                trf_out_shape=output_shape,
            )[0, 0]
            dims = []
            if len(ones_out.shape) == 3:
                dims.append(ones_out.max(1).max(1))
                ones_out_2d = ones_out.max(0)
                dims.append(ones_out_2d.max(1))
                dims.append(ones_out_2d.max(0))
            else:
                raise NotImplementedError

            crop = [(numpy.argmax(dim), numpy.argmax(dim[::-1])) for dim in dims]
            print("determined crop:", crop)

        self.z_offset = crop[0][0]
        self.cropped_output_shape = tuple([outs - sum(c) for outs, c in zip(self.output_shape, crop)])
        self.target_roi = tuple(
            [slice(None), slice(None)] + [slice(c[0], outs - c[1]) for outs, c in zip(self.output_shape, crop)]
        )
        self.order = order

        crop_shift = numpy.eye(len(self.input_shape) + 1, dtype=trf_matrix.dtype)
        crop_shift[:-1, -1] = [c[0] for c in crop]
        self.trf_matrix = trf_matrix.dot(crop_shift)
        self.inv_trf_matrix = numpy.linalg.inv(self.trf_matrix)

    @staticmethod
    def bdv_trafo_to_affine_matrix(trafo):
        """from https://github.com/constantinpape/elf/blob/7b7cd21e632a07876a1302dad92f8d7c1929b37a/elf/transformation/affine.py#L162
        Translate bdv transformation (XYZ) to affine matrix (ZYX)

        """
        assert len(trafo) == 12

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

    @staticmethod
    def concat_affine_matrices(matrices: List[numpy.ndarray]):
        assert all(m.shape == (4, 4) for m in matrices), [m.shape for m in matrices]
        ret = matrices[0]
        for m in matrices[1:]:
            ret = ret.dot(m)

        return ret

    def _impl(
        self,
        ipt: Union[torch.Tensor, numpy.ndarray],
        matrix: numpy.ndarray,
        trf_in_shape: Sequence[int, ...],
        trf_out_shape: Sequence[int, ...],
        select_output_shape: Optional[Sequence[int, ...]] = None,
        z_slices: Optional[Sequence[int]] = None,
    ) -> Union[numpy.ndarray, torch.Tensor]:
        if select_output_shape is None:
            select_output_shape = trf_out_shape

        if isinstance(ipt, numpy.ndarray):
            assert ipt.shape[0] == 1, ipt.shape
            assert ipt.shape[1] == 1, ipt.shape
            ipt = ipt[0, 0]
            if self.input_shape != ipt.shape:
                in_scaling = [trf_in / ipts for ipts, trf_in in zip(ipt.shape, trf_in_shape)] + [1.0]
                matrix = matrix.dot(numpy.diag(in_scaling))

            if self.output_shape != select_output_shape:
                out_scaling = [outs / trf_out for trf_out, outs in zip(trf_out_shape, select_output_shape)] + [1.0]
                matrix = numpy.diag(out_scaling).dot(matrix)

            return affine_transform(ipt, numpy.linalg.inv(matrix), output_shape=select_output_shape, order=self.order)[
                None, None, ...
            ]
        elif isinstance(ipt, torch.Tensor):
            if len(ipt.shape) == 4:
                torch_form = inv_scipy_form2torch_form_2d(
                    matrix,
                    ipt_shape=ipt.shape[2:],
                    trf_in_shape=trf_in_shape,
                    trf_out_shape=trf_out_shape,
                    out_shape=select_output_shape,
                )
            elif len(ipt.shape) == 5:
                torch_form = inv_scipy_form2torch_form_3d(
                    matrix,
                    ipt_shape=ipt.shape[2:],
                    trf_in_shape=trf_in_shape,
                    trf_out_shape=trf_out_shape,
                    out_shape=select_output_shape,
                )
            else:
                raise TypeError(type(ipt))

            # affine_grid_size = tuple(ipt.shape[:2]) + output_shape
            affine_grid_size = (1, 1) + select_output_shape
            if self.affine_grid_size != affine_grid_size:
                self.affine_torch_grid = torch.nn.functional.affine_grid(
                    theta=torch_form, size=affine_grid_size, align_corners=False
                )

            on_cuda = ipt.is_cuda
            ipt_was_cuda = ipt.is_cuda
            if on_cuda != ipt.is_cuda:
                if on_cuda:
                    ipt = ipt.to(torch.device("cuda:0"))
                else:
                    ipt = ipt.to(torch.device("cpu"))

            self.affine_torch_grid = self.affine_torch_grid.to(ipt)

            if z_slices is None:
                affine_grid = self.affine_torch_grid.repeat(ipt.shape[0], ipt.shape[1], 1, 1, 1)
            else:
                assert len(z_slices) == ipt.shape[0], (z_slices, ipt.shape)
                assert all(self.z_offset <= z_slice for z_slice in z_slices), (self.z_offset, z_slices)
                affine_grid = torch.cat(
                    [
                        self.affine_torch_grid[:, z_slice - self.z_offset : z_slice + 1 - self.z_offset]
                        for z_slice in z_slices
                    ]
                )

            ret = torch.nn.functional.grid_sample(
                ipt, affine_grid, align_corners=False, mode=self.mode_from_order[self.order], padding_mode="border"
            )
            if z_slices is not None:
                assert ret.shape[2] == 1
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
        ipt = tensors[self.input_2_transform]
        tgt = tensors[self.target_to_compare_to]

        return self._impl(
            ipt, matrix=self.trf_matrix, trf_in_shape=self.output_shape, trf_out_shape=self.input_shape, **kwargs
        )

    def _forward(self, tensors: OrderedDict[str, Any], **kwargs) -> OrderedDict[str, Any]:
        #         tgt = tensors[self.target_to_compare_to]
        #         assert len(tgt.shape) == 5, tgt.shape
        #         assert tgt.shape[0] == 1, tgt.shape
        #         assert tgt.shape[1] == 1, tgt.shape
        #         assert tgt.shape[2:] == self.output_shape, (tgt.shape, self.output_shape)
        #         tensors[self.target_to_compare_to] = tensors[self.target_to_compare_to][self.target_roi]

        tensors[self.input_2_transform] = self._impl(
            tensors[self.input_2_transform],
            matrix=self.inv_trf_matrix,
            trf_in_shape=self.input_shape,
            trf_out_shape=self.cropped_output_shape,
            z_slices=[m["z_slice"] for m in tensors["meta"]],
            **kwargs,
        )
        return tensors


if __name__ == "__main__":
    trf = AffineTransformation(
        input_2_transform="lr",
        target_to_compare_to="ls",
        order=0,
        input_shape=(838, 1330, 1615),
        output_shape=(241, 1501, 1801),
        matrices=[
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
        crop=[(3, 22), (20, 117), (87, 41)],
    )
