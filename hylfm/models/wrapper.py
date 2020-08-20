from typing import Any, Dict

from lnet import registration
from lnet.models.base import LnetModel


class AffineTransformationAndSliceWrapper(LnetModel):
    def __init__(
        self,
        grid_sampling_scale=(1.0, 1.0, 1.0),
        interpolation_order=2,
        affine_transform_classes: Dict[str, Any] = None,
        **kwargs
    ):
        super().__init__()
        self.grid_sampling_scale = grid_sampling_scale
        affine_transform_classes = affine_transform_classes or {}
        self.affine_transforms = {
            in_shape_for_at: getattr(registration, at_class)(
                order=interpolation_order, trf_out_zoom=grid_sampling_scale
            )
            for in_shape_for_at, at_class in affine_transform_classes.items()
        }
        self.z_dims = {
            in_shape: at.ls_shape[0] - at.lf2ls_crop[0][0] - at.lf2ls_crop[0][1]
            for in_shape, at in self.affine_transforms.items()
        }
        inner_name = kwargs.pop("name")
        assert inner_name != self.__class__.__name__
        from lnet import models

        Inner = getattr(models, inner_name)
        self.inner = Inner(**kwargs)

        self.get_scaling = self.inner.get_scaling
        self.get_shrinkage = self.inner.get_shrinkage
        self.get_output_shape = self.inner.get_output_shape

    def forward(self, x, z_slices=None):
        raise NotImplementedError
        if z_slices is None:
            return self.inner.forward(x)
        else:
            in_shape = ",".join(str(s) for s in x.shape[1:])
            z_dim = int(self.z_dims[in_shape] * self.grid_sampling_scale[0])
            x = self.inner.forward(x)
            out_shape = (z_dim,) + tuple(int(s * g) for s, g in zip(x.shape[3:], self.grid_sampling_scale[1:]))
            return self.affine_transforms[in_shape](x, output_shape=out_shape, z_slices=z_slices)
