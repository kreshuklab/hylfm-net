from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy
import torch.nn.functional
from scipy.ndimage import affine_transform

from lnet.utils.affine import inv_scipy_form2torch_form_2d, inv_scipy_form2torch_form_3d
from scripts.comp import GKRESHUK


class AffineTransform(torch.nn.Module):
    mode_from_order = {0: "nearest", 2: "bilinear"}

    ls_shape: Union[Tuple[int, int], Tuple[int, int, int]]
    lf2ls_crop: Union[Tuple[Tuple[int, int], Tuple[int, int]], Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]]
    lf_shape: Union[Tuple[int, int], Tuple[int, int, int]]
    affine_transforms: Tuple[Tuple[float]]

    def __init__(
        self,
        *,
        forward: str,
        order: int,
        output_shape: Optional[Union[Tuple[int, int], Tuple[int, int, int]]] = None,
        additional_transforms_left: Sequence[numpy.ndarray] = tuple(),
        additional_transforms_right: Sequence[numpy.ndarray] = tuple(),
        lf_shape: Optional[Tuple[int, int, int]] = None,
    ):
        # check if correctly inherited
        # assert hasattr(self, "xml_path") and isinstance(self.xml_path, Path) and self.xml_path.exists(), self.xml_path
        assert hasattr(self, "affine_transforms") and isinstance(self.affine_transforms, tuple), self.affine_transforms
        assert hasattr(self, "ls_shape") and isinstance(self.ls_shape, tuple), self.ls_shape
        assert hasattr(self, "lf2ls_crop") and isinstance(self.lf2ls_crop, tuple), self.lf2ls_crop
        assert hasattr(self, "lf_shape") and isinstance(self.lf_shape, tuple), self.lf_shape

        if forward == "lf2ls":
            self.forward = self.lf2ls
        elif forward == "lsf2lf":
            self.forward = self.ls2lf
        else:
            raise NotImplementedError(forward)

        super().__init__()
        assert output_shape is None or isinstance(output_shape, tuple)
        if output_shape is not None and len(output_shape) == 2:
            raise NotImplementedError

        self.output_shape = output_shape

        if lf_shape is not None:
            self.lf_shape = lf_shape

        self.mode = self.mode_from_order.get(order, None)

        self.trf_matrix = self.concat_affine_matrices(
            list(additional_transforms_left)
            + [self.bdv_trafo_to_affine_matrix(at) for at in self.affine_transforms]
            + list(additional_transforms_right)
        )
        self.inv_trf_matrix = numpy.linalg.inv(self.trf_matrix)
        lf2ls_crop_shift = numpy.eye(len(self.ls_shape) + 1, dtype=self.trf_matrix.dtype)
        lf2ls_crop_shift[:-1, -1] = [c[0] for c in self.lf2ls_crop]
        matrix_with_lf2ls_crop = self.trf_matrix.dot(lf2ls_crop_shift)
        self.lf2ls_matrix = numpy.linalg.inv(matrix_with_lf2ls_crop)
        self.order = order
        self.affine_grid_size = None
        self.z_offset: int = self.lf2ls_crop[0][0]

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

    def _ls2lf(
        self,
        ipt: Union[torch.Tensor, numpy.ndarray],
        matrix: numpy.ndarray,
        trf_in_shape: Optional[Union[Tuple[int, int], Tuple[int, int, int]]] = None,
        trf_out_shape: Optional[Union[Tuple[int, int], Tuple[int, int, int]]] = None,
        output_shape: Optional[Union[Tuple[int, int], Tuple[int, int, int]]] = None,
        order: Optional[int] = None,
        z_slices: Optional[Sequence[int]] = None,
    ) -> Union[numpy.ndarray, torch.Tensor]:
        output_shape = output_shape or self.output_shape
        order = order or self.order
        mode = self.mode_from_order[order]
        if isinstance(ipt, numpy.ndarray):
            assert len(ipt.shape) == 3, ipt.shape
            if trf_in_shape != ipt.shape:
                in_scaling = [trf_in / ipts for ipts, trf_in in zip(ipt.shape, trf_in_shape)] + [1.0]
                matrix = matrix.dot(numpy.diag(in_scaling))

            if trf_out_shape != output_shape:
                out_scaling = [outs / trf_out for trf_out, outs in zip(trf_out_shape, output_shape)] + [1.0]
                matrix = numpy.diag(out_scaling).dot(matrix)

            return affine_transform(ipt, numpy.linalg.inv(matrix), output_shape=output_shape, order=order)
        elif isinstance(ipt, torch.Tensor):
            if len(ipt.shape) == 4:
                torch_form = inv_scipy_form2torch_form_2d(
                    matrix,
                    ipt_shape=ipt.shape[2:],
                    trf_in_shape=trf_in_shape,
                    trf_out_shape=trf_out_shape,
                    out_shape=output_shape,
                )
            elif len(ipt.shape) == 5:
                torch_form = inv_scipy_form2torch_form_3d(
                    matrix,
                    ipt_shape=ipt.shape[2:],
                    trf_in_shape=trf_in_shape,
                    trf_out_shape=trf_out_shape,
                    out_shape=output_shape,
                )
            else:
                raise ValueError(ipt.shape)

            # affine_grid_size = tuple(ipt.shape[:2]) + output_shape
            affine_grid_size = (1, 1) + output_shape
            if self.affine_grid_size != affine_grid_size:
                self.affine_torch_grid = torch.nn.functional.affine_grid(
                    theta=torch_form, size=affine_grid_size, align_corners=False
                )

            on_cuda = ipt.is_cuda
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
                ipt, affine_grid, align_corners=False, mode=mode, padding_mode="border"
            )
            if z_slices is not None:
                assert ret.shape[2] == 1
                ret = ret[:, :, 0]

            if on_cuda == ipt.is_cuda:
                return ret
            elif ipt.is_cuda:
                return ret.to(device=torch.device("cuda:0"))
            else:
                return ret.to(device=torch.device("cpu"))
        else:
            raise TypeError(type(ipt))

    def ls2lf(self, ipt: Union[torch.Tensor, numpy.ndarray], **kwargs) -> Union[numpy.ndarray, torch.Tensor]:
        return self._ls2lf(
            ipt, matrix=self.trf_matrix, trf_in_shape=self.ls_shape, trf_out_shape=self.lf_shape, **kwargs
        )

    def lf2ls(self, ipt: Union[torch.Tensor, numpy.ndarray], **kwargs) -> Union[numpy.ndarray, torch.Tensor]:
        # cropped_ls_shape: Union[Tuple[int, int], Tuple[int, int, int]] = tuple(
        #     (lss - ls_crop[0] - ls_crop[1]) * zoom
        #     for lss, ls_crop, zoom in zip(self.ls_shape, self.lf2ls_crop, self.trf_out_zoom)
        # )
        cropped_ls_shape = tuple(
            (lss - ls_crop[0] - ls_crop[1])
            for lss, ls_crop in zip(self.ls_shape, self.lf2ls_crop)
        )
        return self._ls2lf(
            ipt, matrix=self.lf2ls_matrix, trf_in_shape=self.lf_shape, trf_out_shape=cropped_ls_shape, **kwargs
        )
