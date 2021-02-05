from hylfm.datasets.named import DatasetPart
from hylfm.hylfm_types import DatasetName, TransformsPipeline
from hylfm.transforms import (
    AdditiveGaussianNoise,
    Assert,
    Cast,
    ChannelFromLightField,
    ComposedTransform,
    Crop,
    CropWhatShrinkDoesNot,
    Normalize01Dataset,
    NormalizeMSE,
    PoissonNoise,
    RandomIntensityScale,
    RandomRotate90,
    RandomlyFlipAxis,
)


def get_transforms_pipeline(
    dataset_name: DatasetName,
    dataset_part: DatasetPart,
    nnum: int,
    z_out: int,
    scale: int,
    shrink: int,
    interpolation_order: int = 2,
):
    meta = {
        "nnum": nnum,
        "z_out": z_out,
        "scale": scale,
        "interpolation_order": interpolation_order,
        "crop_names": set(),
    }
    if dataset_name in [DatasetName.beads_highc_a]:
        spim = "ls_reg"
        crop_names = []
        if dataset_name == DatasetName.beads_highc_a and dataset_part == DatasetPart.train:
            if scale != 8:
                # due to size the zenodo upload is resized to scale 8
                sample_precache_trf = [
                    {"Resize": {"apply_to": spim, "shape": [1.0, 121, scale / 8, scale / 8], "order": 2}}
                ]
            else:
                sample_precache_trf = []
        else:
            sample_precache_trf = [
                {"Resize": {"apply_to": spim, "shape": [1.0, 121, scale / 19, scale / 19], "order": 2}}
            ]

        sample_precache_trf.append({"Assert": {"apply_to": spim, "expected_tensor_shape": [None, 1, 121, None, None]}})

        if dataset_part == DatasetPart.train:
            sample_preprocessing = ComposedTransform(
                Crop(apply_to=spim, crop=((0, None), (35, -35), (shrink, -shrink), (shrink, -shrink))),
                Normalize01Dataset(apply_to="lf", min_percentile=5.0, max_percentile=99.8),
                Normalize01Dataset(apply_to=spim, min_percentile=5.0, max_percentile=99.99),
                AdditiveGaussianNoise(apply_to="lf", sigma=0.1),
                AdditiveGaussianNoise(apply_to=spim, sigma=0.05),
                RandomIntensityScale(apply_to=["lf", spim], factor_min=0.8, factor_max=1.2, independent=False),
                RandomlyFlipAxis(apply_to=["lf", spim], axis=-1),
                RandomlyFlipAxis(apply_to=["lf", spim], axis=-2),
            )
            batch_preprocessing = ComposedTransform(
                RandomRotate90(apply_to=["lf", spim]), ChannelFromLightField(apply_to={"lf": "lfc"}, nnum=nnum)
            )
        else:
            sample_preprocessing = ComposedTransform(
                Crop(apply_to=spim, crop=((0, None), (35, -35), (shrink, -shrink), (shrink, -shrink))),
                Normalize01Dataset(apply_to="lf", min_percentile=5.0, max_percentile=99.8),
                Normalize01Dataset(apply_to=spim, min_percentile=5.0, max_percentile=99.99),
                ChannelFromLightField(apply_to={"lf": "lfc"}, nnum=nnum),
            )
            batch_preprocessing = ComposedTransform()

        batch_preprocessing_in_step = Cast(apply_to=["lfc", spim], dtype="float32", device="cuda", non_blocking=True)
        batch_postprocessing = ComposedTransform(
            Assert(apply_to="pred", expected_tensor_shape=(None, 1, z_out, None, None)),
            Assert(apply_to="pred", expected_shape_like_tensor=spim),
        )

    elif dataset_name in [DatasetName.beads_sample0, DatasetName.beads_highc_b]:
        spim = "ls_reg"
        crop_names = []
        if dataset_name == DatasetName.beads_highc_a and dataset_part == DatasetPart.train:
            if scale != 8:
                # due to size the zenodo upload is resized to scale 8
                sample_precache_trf = [
                    {"Resize": {"apply_to": spim, "shape": [1.0, 121, scale / 8, scale / 8], "order": 2}}
                ]
            else:
                sample_precache_trf = []
        else:
            sample_precache_trf = [
                {"Resize": {"apply_to": spim, "shape": [1.0, 121, scale / 19, scale / 19], "order": 2}}
            ]

        sample_precache_trf.append({"Assert": {"apply_to": spim, "expected_tensor_shape": [None, 1, 121, None, None]}})

        if dataset_part == DatasetPart.train:
            sample_preprocessing = ComposedTransform(
                Crop(apply_to=spim, crop=((0, None), (35, -35), (shrink, -shrink), (shrink, -shrink))),
                Normalize01Dataset(apply_to="lf", min_percentile=5.0, max_percentile=99.8),
                Normalize01Dataset(apply_to=spim, min_percentile=5.0, max_percentile=99.95),
                AdditiveGaussianNoise(apply_to="lf", sigma=0.1),
                AdditiveGaussianNoise(apply_to=spim, sigma=0.05),
                RandomIntensityScale(apply_to=["lf", spim], factor_min=0.8, factor_max=1.2, independent=False),
                RandomlyFlipAxis(apply_to=["lf", spim], axis=-1),
                RandomlyFlipAxis(apply_to=["lf", spim], axis=-2),
            )
            batch_preprocessing = ComposedTransform(
                RandomRotate90(apply_to=["lf", spim]), ChannelFromLightField(apply_to={"lf": "lfc"}, nnum=nnum)
            )
        else:
            sample_preprocessing = ComposedTransform(
                Crop(apply_to=spim, crop=((0, None), (35, -35), (shrink, -shrink), (shrink, -shrink))),
                Normalize01Dataset(apply_to="lf", min_percentile=5.0, max_percentile=99.8),
                Normalize01Dataset(apply_to=spim, min_percentile=5.0, max_percentile=99.95),
                ChannelFromLightField(apply_to={"lf": "lfc"}, nnum=nnum),
            )
            batch_preprocessing = ComposedTransform()

        batch_preprocessing_in_step = Cast(apply_to=["lfc", spim], dtype="float32", device="cuda", non_blocking=True)
        batch_postprocessing = ComposedTransform(
            Assert(apply_to="pred", expected_tensor_shape=(None, 1, z_out, None, None)),
            Assert(apply_to="pred", expected_shape_like_tensor=spim),
        )

    elif dataset_name in [
        DatasetName.heart_static_sample0,
        DatasetName.heart_static_a,
        DatasetName.heart_static_b,
        DatasetName.heart_static_c,
    ]:
        spim = "ls_trf"
        crop_names = ["staticHeartFOV", "Heart_tightCrop"]
        sample_precache_trf = []

        if dataset_name in [DatasetName.heart_static_c, DatasetName.heart_static_sample0]:
            spim_max_percentile = 99.8
        else:
            spim_max_percentile = 99.99

        if dataset_part == DatasetPart.train:
            sample_preprocessing = ComposedTransform(
                CropWhatShrinkDoesNot(
                    apply_to="lf", nnum=nnum, scale=scale, shrink=shrink, wrt_ref=True, crop_names=crop_names
                ),
                CropWhatShrinkDoesNot(
                    apply_to=spim, nnum=nnum, scale=scale, shrink=shrink, wrt_ref=False, crop_names=crop_names
                ),
                Crop(apply_to=spim, crop=((0, None), (0, None), (shrink, -shrink), (shrink, -shrink))),
                Normalize01Dataset(apply_to="lf", min_percentile=5.0, max_percentile=99.8),
                Normalize01Dataset(apply_to=spim, min_percentile=5.0, max_percentile=spim_max_percentile),
                RandomIntensityScale(apply_to=["lf", spim], factor_min=0.8, factor_max=1.2, independent=False),
                PoissonNoise(apply_to="lf", peak=10),
                PoissonNoise(apply_to=spim, peak=10),
                RandomlyFlipAxis(apply_to=["lf", spim], axis=-1),
                RandomlyFlipAxis(apply_to=["lf", spim], axis=-2),
            )
            batch_preprocessing = ComposedTransform(
                RandomRotate90(apply_to=["lf", spim]), ChannelFromLightField(apply_to={"lf": "lfc"}, nnum=nnum)
            )
        else:
            sample_preprocessing = ComposedTransform(
                CropWhatShrinkDoesNot(
                    apply_to="lf", nnum=nnum, scale=scale, shrink=shrink, wrt_ref=True, crop_names=crop_names
                ),
                CropWhatShrinkDoesNot(
                    apply_to=spim, nnum=nnum, scale=scale, shrink=shrink, wrt_ref=False, crop_names=crop_names
                ),
                Crop(apply_to=spim, crop=((0, None), (0, None), (shrink, -shrink), (shrink, -shrink))),
                Normalize01Dataset(apply_to="lf", min_percentile=5.0, max_percentile=99.8),
                Normalize01Dataset(apply_to=spim, min_percentile=5.0, max_percentile=spim_max_percentile),
                ChannelFromLightField(apply_to={"lf": "lfc"}, nnum=nnum),
            )
            batch_preprocessing = ComposedTransform()

        batch_preprocessing_in_step = Cast(apply_to=["lfc", spim], dtype="float32", device="cuda", non_blocking=True)
        batch_postprocessing = ComposedTransform(
            Assert(apply_to="pred", expected_tensor_shape=(None, 1, z_out, None, None)),
            Assert(apply_to="pred", expected_shape_like_tensor=spim),
        )

    else:
        raise NotImplementedError(dataset_name)

    meta["crop_names"].update(set(crop_names))
    batch_premetric_trf = ComposedTransform(NormalizeMSE(apply_to="pred", target_name=spim, return_alpha_beta=True))
    return TransformsPipeline(
        sample_precache_trf=sample_precache_trf,
        sample_preprocessing=sample_preprocessing,
        batch_preprocessing=batch_preprocessing,
        batch_preprocessing_in_step=batch_preprocessing_in_step,
        batch_postprocessing=batch_postprocessing,
        batch_premetric_trf=batch_premetric_trf,
        meta=meta,
    )
