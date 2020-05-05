import logging
from typing import Any, List, Optional, OrderedDict, Sequence, Tuple, Union, Dict

import numpy
import torch.nn.functional
import torch.nn.functional
from scipy.ndimage import affine_transform

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
    scipy_padding_mode = {"zeros": "constant", "border": "nearest"}

    def __init__(
        self,
        *,
        apply_to: Union[str, Dict[str, str]],
        target_to_compare_to: Union[str, Tuple[int, int, int]],
        order: int,
        ref_input_shape: Sequence[int],
        bdv_affine_transformations: List[
            List[float]
        ],  # Fiij's big data viewer affine transformations: each affine transformation as a list of 12 floats.
        # affine_matrices: List[List[float]],
        ref_output_shape: Sequence[int],
        ref_crop_in: Optional[Tuple[Tuple[int, int], ...]] = None,
        ref_crop_out: Optional[Tuple[Tuple[int, int], ...]] = None,
        inverted: bool = False,
        padding_mode: str = "border",
    ):
        if len(ref_input_shape) not in (2, 3):
            raise NotImplementedError

        if len(ref_output_shape) not in (2, 3):
            raise NotImplementedError

        super().__init__()
        if isinstance(apply_to, str):
            apply_to = {apply_to: apply_to}

        self.apply_to: Dict[str, str] = apply_to
        self.target_to_compare_to = target_to_compare_to
        self.input_shape = ref_input_shape
        self.output_shape = ref_output_shape

        self.mode = self.mode_from_order[order]
        self.padding_mode = padding_mode

        assert len(bdv_affine_transformations) >= 1
        trf_matrices = [bdv_trafo_to_affine_matrix(m) for m in bdv_affine_transformations]
        trf_matrix = trf_matrices[0]
        for m in trf_matrices[1:]:
            trf_matrix = trf_matrix.dot(m)

        self.affine_grids = {}

        self.forward = self._inverted if inverted else self._forward

        if ref_crop_in is None:
            ref_crop_in = tuple([(0, 0) for _ in range(len(ref_input_shape))])
        else:
            assert len(ref_crop_in) == len(ref_input_shape)
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

            logger.info("determined crop_out: %s", ref_crop_out)
        elif len(ref_crop_out) == len(ref_output_shape) + 1:
            assert ref_crop_out[0][0] == 0 and ref_crop_out[0][1] is None or ref_crop_out[0][1] == 0, ref_crop_out
            ref_crop_out = ref_crop_out[1:]

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
                assert affine_grid.shape[0] == 1
                affine_grid = torch.cat(
                    [affine_grid[:, z_slice - self.z_offset : z_slice + 1 - self.z_offset] for z_slice in z_slices]
                )

            ret = torch.nn.functional.grid_sample(
                ipt, affine_grid, align_corners=False, mode=self.mode, padding_mode=self.padding_mode
            )

            if on_cuda == ipt_was_cuda:
                return ret
            elif ipt_was_cuda:
                return ret.to(device=torch.device("cuda:0"))
            else:
                return ret.to(device=torch.device("cpu"))
        else:
            raise TypeError(type(ipt))

    def _inverted(self, tensors: OrderedDict[str, Any]) -> OrderedDict[str, Any]:
        if isinstance(self.target_to_compare_to, str):
            z_slices = [m.get(self.target_to_compare_to, {}).get("z_slice", None) for m in tensors["meta"]]
            output_sampling_shape = tensors[self.target_to_compare_to].shape[2:]
        else:
            z_slices = None
            output_sampling_shape = tuple(self.target_to_compare_to)

        assert 0 not in output_sampling_shape, output_sampling_shape
        assert len(output_sampling_shape) == 3, output_sampling_shape

        if output_sampling_shape[0] == 1:
            # single z_slice:
            output_sampling_shape = (self.cropped_input_shape[0],) + output_sampling_shape[1:]

        for in_name, out_name in self.apply_to.items():
            tensors[out_name] = self._impl(
                ipt=tensors[in_name],
                matrix=self.trf_matrix_inv,
                trf_in_shape=self.cropped_output_shape,
                trf_out_shape=self.cropped_input_shape,
                order=self.order,
                output_sampling_shape=output_sampling_shape,
                z_slices=z_slices,
            )
        return tensors

    def _forward(self, tensors: OrderedDict[str, Any]) -> OrderedDict[str, Any]:
        if isinstance(self.target_to_compare_to, str):
            z_slices = [m.get(self.target_to_compare_to, {}).get("z_slice", None) for m in tensors["meta"]]
            output_sampling_shape = tensors[self.target_to_compare_to].shape[2:]
        else:
            z_slices = None
            output_sampling_shape = tuple(self.target_to_compare_to)

        assert 0 not in output_sampling_shape, output_sampling_shape
        assert len(output_sampling_shape) == 3, output_sampling_shape

        if output_sampling_shape[0] == 1:
            # single z_slice:
            output_sampling_shape = (self.cropped_output_shape[0],) + output_sampling_shape[1:]

        for in_name, out_name in self.apply_to.items():
            tensors[out_name] = self._impl(
                ipt=tensors[in_name],
                matrix=self.trf_matrix,
                trf_in_shape=self.cropped_input_shape,
                trf_out_shape=self.cropped_output_shape,
                order=self.order,
                output_sampling_shape=output_sampling_shape,
                z_slices=z_slices,
            )
        return tensors


def static():
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt

    from lnet.datasets import (
        get_dataset_from_info,
        ZipDataset,
        N5CachedDatasetFromInfo,
        get_collate_fn,
        N5CachedDatasetFromInfoSubset,
    )
    from lnet.datasets.gcamp import ref0_lr, ref0_ls
    from lnet.transformations import ComposedTransformation, Cast, Crop

    ref0_lr.transformations += [
        {
            "Resize": {
                "apply_to": "lr",
                "shape": [1.0, 1.0, 0.21052631578947368421052631578947, 0.21052631578947368421052631578947],
                "order": 2,
            }  # 4/19=0.21052631578947368421052631578947; 8/19=0.42105263157894736842105263157895
        }
    ]
    ref0_ls.transformations += [
        {
            "Resize": {
                "apply_to": "ls",
                "shape": [1.0, 1.0, 0.21052631578947368421052631578947, 0.21052631578947368421052631578947],
                "order": 2,
            }  # 4/19=0.21052631578947368421052631578947; 8/19=0.42105263157894736842105263157895
        }
    ]
    lrds = N5CachedDatasetFromInfoSubset(
        N5CachedDatasetFromInfo(get_dataset_from_info(ref0_lr)), indices=[0], filters=[]
    )
    lsds = N5CachedDatasetFromInfoSubset(
        N5CachedDatasetFromInfo(get_dataset_from_info(ref0_ls)), indices=[0], filters=[]
    )
    trf = ComposedTransformation(
        Crop(apply_to="lr", crop=((0, None), (12, -12), (28, 236), (68, 336))),
        # Crop(apply_to="ls", crop=((0, None), (60, -60), (0, None), (0, None))),
        Crop(apply_to="ls", crop=((0, None), (60, -60), (32, -72), (92, -20))),
        Cast(apply_to={"lr": "lr_torch", "ls": "ls_torch"}, dtype="float32", device="cuda"),
    )
    ds = ZipDataset({"lr": lrds, "ls": lsds}, transformation=trf)
    loader = DataLoader(ds, batch_size=1, collate_fn=get_collate_fn(lambda t: t))

    assert torch.cuda.device_count() == 1, torch.cuda.device_count()
    sample = next(iter(loader))
    lr = sample["lr"]
    print("lr", lr.shape)
    ls = sample["ls"]
    print("ls", ls.shape)
    # trf = AffineTransformation(apply_to="lr", target_to_compare_to="ls", order=0, input_shape=(838, 1330, 1615), output_shape=(241, 1501, 1801), matrices=[[0.98048,0.004709,0.098297,-111.7542,7.6415e-05,0.97546,0.0030523,-20.1143,0.014629,8.2964e-06,-3.9928,846.8515]], crop_in=[(5, -5), (10, -10), (10, -10)], crop_out=[(3, -22), (20, -117), (87, -41)])
    # ref_crop_in = ((208 + 0, -208-14), (133, 1121), (323, 1596))  # 323, 133, 1273, 988
    # ref_crop_in = ((208-11, -208+49), (133-20, 1121), (323, 1596))  # 323, 133, 1273, 988
    ref_crop_in = ((208 - 56, -208 - 15), (133, 1121), (323, 1596))  # 323, 133, 1273, 988
    # ref_crop_out = ((60 + 0, -60-19), (152, -323), (418, -57))
    # ref_crop_out = ((60, -60), (0, 0), (0, 0))
    # ref_crop_out = ((60, -60), (157, -332-7), (421+10, -67-15))
    ref_crop_out = ((60, -60), (152, -342), (437, -95))
    # ref_crop_out = None
    # trf_torch = AffineTransformation(
    #     apply_to="lr_torch",
    #     target_to_compare_to="ls_torch",
    #     order=0,
    #     ref_input_shape=(838, 1330, 1615),
    #     ref_output_shape=(241, 1501, 1801),
    #     bdv_affine_transformations=[
    #         [
    #             0.98048,
    #             0.004709,
    #             0.098297,
    #             -111.7542,
    #             7.6415e-05,
    #             0.97546,
    #             0.0030523,
    #             -20.1143,
    #             0.014629,
    #             8.2964e-06,
    #             -3.9928,
    #             846.8515,
    #         ]
    #     ],
    #     ref_crop_in=ref_crop_in,
    #     ref_crop_out=ref_crop_out,
    # )
    #
    trf_numpy = AffineTransformation(
        apply_to="lr",
        target_to_compare_to="ls",
        order=0,
        ref_input_shape=(838, 1330, 1615),
        ref_output_shape=(241, 1501, 1801),
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
        ref_crop_in=ref_crop_in,
        ref_crop_out=ref_crop_out,
        padding_mode="zeros",
    )

    one_sample = {"lr": numpy.ones((1, 1, 100, 100, 100)), "ls": sample["ls"], "meta": [{}]}
    one_sample = trf_numpy(one_sample)
    ones = one_sample["lr"][0, 0]
    fig, ax = plt.subplots(ncols=3, figsize=(30, 10))
    plt.title("ls")
    for i in range(3):
        img = ones.max(i)
        ax[i].imshow(img)

    plt.show()
    return

    sample = trf_numpy(sample)
    sample = trf_torch(sample)
    out = sample["lr"][0, 0]
    out_torch = sample["lr_torch"][0, 0].cpu().numpy()
    ls = sample["ls"][0, 0]
    print("ls, lr_numpy, lr_torch", ls.shape, out.shape, out_torch.shape)

    fig, ax = plt.subplots(ncols=3, figsize=(30, 10))
    plt.title("ls")
    for i in range(3):
        img = ls.max(i)
        ax[i].imshow(img)

    plt.show()

    fig, ax = plt.subplots(ncols=3, figsize=(30, 10))
    plt.title("numpy")
    for i in range(3):
        img = out.max(i)
        ax[i].imshow(img)

    plt.show()

    fig, ax = plt.subplots(ncols=3, figsize=(30, 10))
    plt.title("torch")
    for i in range(3):
        img = out_torch.max(i)
        ax[i].imshow(img)

    plt.show()

    fig, ax = plt.subplots(ncols=3, figsize=(30, 10))
    plt.title("numpy vs torch")
    for i in range(3):
        out_img = out.max(i)
        out_torch_img = out_torch.max(i)
        img = numpy.zeros_like(out_img)
        img[::2, ::2] = out_img[::2, ::2]
        img[1::2, 1::2] = out_img[1::2, 1::2]
        img[::2, 1::2] = -out_torch_img[::2, 1::2]
        img[1::2, ::2] = -out_torch_img[1::2, ::2]
        ax[i].imshow(img)

    plt.show()


def dynamic():
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt

    from lnet.datasets import (
        get_dataset_from_info,
        ZipDataset,
        N5CachedDatasetFromInfo,
        get_collate_fn,
        N5CachedDatasetFromInfoSubset,
    )
    from lnet.datasets.gcamp import ref0_lr, ref0_sample_ls_slice
    from lnet.transformations import ComposedTransformation, Cast, Crop, Assert

    ref0_lr.transformations += [
        {
            "Resize": {
                "apply_to": "lr",
                "shape": [1.0, 1.0, 0.21052631578947368421052631578947, 0.21052631578947368421052631578947],
                "order": 2,
            }  # 4/19=0.21052631578947368421052631578947; 8/19=0.42105263157894736842105263157895
        }
    ]
    ref0_sample_ls_slice.transformations += [
        {"Assert": {"apply_to": "ls", "expected_tensor_shape": [1, 1, 1, None, None]}},
        {
            "Resize": {
                "apply_to": "ls",
                "shape": [1.0, 1.0, 0.21052631578947368421052631578947, 0.21052631578947368421052631578947],
                "order": 2,
            }  # 4/19=0.21052631578947368421052631578947; 8/19=0.42105263157894736842105263157895
        },
    ]
    lrds = N5CachedDatasetFromInfoSubset(
        N5CachedDatasetFromInfo(get_dataset_from_info(ref0_lr)), indices=[0], filters=[]
    )
    lsds = N5CachedDatasetFromInfoSubset(
        N5CachedDatasetFromInfo(get_dataset_from_info(ref0_sample_ls_slice)), indices=[110], filters=[]
    )
    trf = ComposedTransformation(
        Assert(apply_to="ls", expected_tensor_shape=(1, 1, 1, None, None)),
        Assert(apply_to="lr", expected_tensor_shape=(1, 1, None, None, None)),
        Crop(apply_to="lr", crop=((0, None), (0, None), (28, 236), (68, 336))),
        Crop(apply_to="ls", crop=((0, None), (0, None), (0, None), (0, None))),
        Crop(apply_to="ls", crop=((0, None), (0, None), (32, -68), (88, -12))),
        Cast(apply_to={"lr": "lr_torch", "ls": "ls_torch"}, dtype="float32", device="cuda"),
    )
    ds = ZipDataset({"lr": lrds, "ls": lsds}, transformation=trf, join_dataset_masks=False)
    loader = DataLoader(ds, batch_size=1, collate_fn=get_collate_fn(lambda t: t))
    assert torch.cuda.device_count() == 1, torch.cuda.device_count()

    sample = next(iter(loader))
    lr = sample["lr"]
    print("lr", lr.shape)
    ls = sample["ls"]
    print("ls", ls.shape)
    # trf = AffineTransformation(apply_to="lr", target_to_compare_to="ls", order=0, input_shape=(838, 1330, 1615), output_shape=(241, 1501, 1801), matrices=[[0.98048,0.004709,0.098297,-111.7542,7.6415e-05,0.97546,0.0030523,-20.1143,0.014629,8.2964e-06,-3.9928,846.8515]], crop_in=[(5, -5), (10, -10), (10, -10)], crop_out=[(3, -22), (20, -117), (87, -41)])
    ref_crop_in = ((-17 + 208, +76 - 208), (133, 1121), (323, 1596))  # 323, 133, 1273, 988
    ref_crop_out = ((0 + 60, -0 - 60), (152, -323), (418, -57))
    # ref_crop_out = ((0, 0), (0, 0), (0, 0))
    trf_torch = AffineTransformation(
        apply_to="lr_torch",
        target_to_compare_to="ls_torch",
        order=0,
        ref_input_shape=(838, 1330, 1615),
        ref_output_shape=(241, 1501, 1801),
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
        ref_crop_in=ref_crop_in,
        ref_crop_out=ref_crop_out,
    )

    trf_numpy = AffineTransformation(
        apply_to="lr",
        target_to_compare_to="ls",
        order=0,
        ref_input_shape=(838, 1330, 1615),
        ref_output_shape=(241, 1501, 1801),
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
        ref_crop_in=ref_crop_in,
        ref_crop_out=ref_crop_out,
    )

    sample = trf_numpy(sample)
    sample = trf_torch(sample)
    out = sample["lr"][0, 0]
    assert out.shape[0] == 1, out.shape
    out = out[0]
    out_torch = sample["lr_torch"][0, 0].cpu().numpy()
    assert out_torch.shape[0] == 1, out_torch.shape
    out_torch = out_torch[0]
    ls = sample["ls"][0, 0]
    assert ls.shape[0] == 1, ls.shape
    ls = ls[0]
    print("ls, lr_numpy, lr_torch", ls.shape, out.shape, out_torch.shape)

    plt.figure(figsize=(30, 10))
    plt.imshow(ls)
    plt.title("ls")
    plt.show()

    plt.figure(figsize=(30, 10))
    plt.imshow(out)
    plt.title("numpy")
    plt.show()

    plt.figure(figsize=(30, 10))
    plt.imshow(out_torch)
    plt.title("torch")
    plt.show()

    plt.figure(figsize=(30, 10))
    img = numpy.zeros_like(out)
    img[::2, ::2] = out[::2, ::2]
    img[1::2, 1::2] = out[1::2, 1::2]
    img[::2, 1::2] = -out_torch[::2, 1::2]
    img[1::2, ::2] = -out_torch[1::2, ::2]
    plt.imshow(img)
    plt.title("numpy vs torch")
    plt.show()


if __name__ == "__main__":
    static()
