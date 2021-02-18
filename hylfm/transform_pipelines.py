from typing import Tuple, Union

from hylfm.datasets.named import DatasetPart
from hylfm.hylfm_types import DatasetChoice, TransformsPipeline
from hylfm.transforms import (
    AdditiveGaussianNoise,
    AffineTransformationDynamicTraining,
    Assert,
    Cast,
    ChannelFromLightField,
    ComposedTransform,
    Crop,
    CropLSforDynamicTraining,
    CropWhatShrinkDoesNot,
    Identity,
    Normalize01Dataset,
    NormalizeMSE,
    PoissonNoise,
    RandomIntensityScale,
    RandomRotate90,
    RandomlyFlipAxis,
)


def get_transforms_pipeline(
    dataset_name: DatasetChoice,
    dataset_part: DatasetPart,
    nnum: int,
    z_out: int,
    scale: int,
    shrink: int,
    interpolation_order: int = 2,
    incl_pred_vol: bool = False,
    load_lfd_and_care: bool = False,
):
    sliced = dataset_name.value.endswith("_sliced") and dataset_part == DatasetPart.train
    dynamic = "dyn" in dataset_name.value
    spatial_dims = 2 if sliced or dynamic else 3
    dyn_precomputed = dataset_name in [DatasetChoice.heart_dyn_refine_lfd]

    pred_z_min = 0
    pred_z_max = 838
    z_ls_rescaled = 241
    crop_names = set()

    if dataset_name == DatasetChoice.heart_static_mix3_sliced:
        crop_names.add("heart_2020_02_fish1_static")
        crop_names.add("Heart_tightCrop")
        crop_names.add("staticHeartFOV")

        sample_precache_trf = []

        tgt_max_percentile = 99.8

        sample_preprocessing = ComposedTransform(
            CropWhatShrinkDoesNot(
                apply_to="lf", nnum=nnum, scale=scale, shrink=shrink, wrt_ref=True, crop_names=crop_names
            )
        )

        if sliced or dynamic:
            tgt = "ls_slice"
            sample_preprocessing += CropLSforDynamicTraining(
                apply_to=tgt, crop_names=crop_names, nnum=nnum, scale=scale, z_ls_rescaled=z_ls_rescaled
            )
        else:
            tgt = "ls_trf"
            sample_preprocessing += CropWhatShrinkDoesNot(
                apply_to=tgt, nnum=nnum, scale=scale, shrink=shrink, wrt_ref=False, crop_names=crop_names
            )

            sample_preprocessing += Crop(
                apply_to=tgt, crop=((0, None), (0, None), (shrink, -shrink), (shrink, -shrink))
            )

        sample_preprocessing += Normalize01Dataset(apply_to="lf", min_percentile=5.0, max_percentile=99.8)
        sample_preprocessing += Normalize01Dataset(apply_to=tgt, min_percentile=5.0, max_percentile=tgt_max_percentile)

        if dataset_part == DatasetPart.train:
            sample_preprocessing += ComposedTransform(
                RandomIntensityScale(apply_to=["lf", tgt], factor_min=0.8, factor_max=1.2, independent=False),
                PoissonNoise(apply_to="lf", peak=10),
                PoissonNoise(apply_to=tgt, peak=10),
                # AdditiveGaussianNoise(apply_to="lf", sigma=0.1),
                # AdditiveGaussianNoise(apply_to=tgt, sigma=0.05),
                RandomlyFlipAxis(apply_to=["lf", tgt], axis=-1),
                RandomlyFlipAxis(apply_to=["lf", tgt], axis=-2),
            )
            batch_preprocessing = ComposedTransform(
                RandomRotate90(apply_to=["lf", tgt]), ChannelFromLightField(apply_to={"lf": "lfc"}, nnum=nnum)
            )
        else:
            sample_preprocessing += ComposedTransform(ChannelFromLightField(apply_to={"lf": "lfc"}, nnum=nnum))
            batch_preprocessing = ComposedTransform()

        batch_preprocessing_in_step = Cast(apply_to=["lfc", tgt], dtype="float32", device="cuda", non_blocking=True)
        batch_postprocessing = ComposedTransform(
            Assert(apply_to="pred", expected_tensor_shape=(None, 1, z_out, None, None))
        )

    elif dataset_name in [DatasetChoice.heart_static_mix1_sliced, DatasetChoice.heart_static_mix2_sliced]:
        crop_names.add("heart_2020_02_fish1_static")
        crop_names.add("heart_2020_02_fish2_static")

        spim_max_percentile = 99.8

        sample_precache_trf = []

        sample_preprocessing = ComposedTransform(
            CropWhatShrinkDoesNot(
                apply_to="lf", nnum=nnum, scale=scale, shrink=shrink, wrt_ref=True, crop_names=crop_names
            )
        )
        if load_lfd_and_care:
            sample_preprocessing += CropWhatShrinkDoesNot(
                apply_to="lfd", nnum=nnum, scale=scale, shrink=shrink, wrt_ref=False, crop_names=crop_names
            )

        if sliced or dynamic:
            tgt = "ls_slice"
            sample_preprocessing += CropLSforDynamicTraining(
                apply_to=tgt, crop_names=crop_names, nnum=nnum, scale=scale, z_ls_rescaled=z_ls_rescaled
            )
        else:
            tgt = "ls_trf"
            sample_preprocessing += CropWhatShrinkDoesNot(
                apply_to=tgt, nnum=nnum, scale=scale, shrink=shrink, wrt_ref=False, crop_names=crop_names
            )

            sample_preprocessing += Crop(
                apply_to=tgt, crop=((0, None), (0, None), (shrink, -shrink), (shrink, -shrink))
            )

        sample_preprocessing += Normalize01Dataset(apply_to="lf", min_percentile=5.0, max_percentile=99.8)
        sample_preprocessing += Normalize01Dataset(apply_to=tgt, min_percentile=5.0, max_percentile=spim_max_percentile)

        if dataset_part == DatasetPart.train:
            sample_preprocessing += ComposedTransform(
                RandomIntensityScale(apply_to=["lf", tgt], factor_min=0.8, factor_max=1.2, independent=False),
                PoissonNoise(apply_to="lf", peak=10),
                PoissonNoise(apply_to=tgt, peak=10),
                # AdditiveGaussianNoise(apply_to="lf", sigma=0.1),
                # AdditiveGaussianNoise(apply_to=spim, sigma=0.05),
                RandomlyFlipAxis(apply_to=["lf", tgt], axis=-1),
                RandomlyFlipAxis(apply_to=["lf", tgt], axis=-2),
            )
            batch_preprocessing = ComposedTransform(
                RandomRotate90(apply_to=["lf", tgt]), ChannelFromLightField(apply_to={"lf": "lfc"}, nnum=nnum)
            )
        else:
            sample_preprocessing += ComposedTransform(ChannelFromLightField(apply_to={"lf": "lfc"}, nnum=nnum))
            batch_preprocessing = ComposedTransform()

        batch_preprocessing_in_step = ComposedTransform(
            Cast(apply_to=["lfc", tgt], dtype="float32", device="cuda", non_blocking=True)
        )
        if load_lfd_and_care:
            batch_preprocessing_in_step += Cast(apply_to=["lfd"], dtype="float32", device="cuda", non_blocking=True)

        batch_postprocessing = ComposedTransform(
            Assert(apply_to="pred", expected_tensor_shape=(None, 1, z_out, None, None))
        )

    elif dataset_name in [DatasetChoice.beads_highc_a]:
        tgt = "ls_reg"
        if dataset_name == DatasetChoice.beads_highc_a and dataset_part == DatasetPart.train:
            if scale != 8:
                # due to size the zenodo upload is resized to scale 8
                sample_precache_trf = [
                    {"Resize": {"apply_to": tgt, "shape": [1.0, 121, scale / 8, scale / 8], "order": 2}}
                ]
            else:
                sample_precache_trf = []
        else:
            sample_precache_trf = [
                {"Resize": {"apply_to": tgt, "shape": [1.0, 121, scale / 19, scale / 19], "order": 2}}
            ]

        sample_precache_trf.append({"Assert": {"apply_to": tgt, "expected_tensor_shape": [None, 1, 121, None, None]}})

        if dataset_part == DatasetPart.train:
            sample_preprocessing = ComposedTransform(
                Crop(apply_to=tgt, crop=((0, None), (35, -35), (shrink, -shrink), (shrink, -shrink))),
                Normalize01Dataset(apply_to="lf", min_percentile=5.0, max_percentile=99.8),
                Normalize01Dataset(apply_to=tgt, min_percentile=5.0, max_percentile=99.99),
                AdditiveGaussianNoise(apply_to="lf", sigma=0.1),
                AdditiveGaussianNoise(apply_to=tgt, sigma=0.05),
                RandomIntensityScale(apply_to=["lf", tgt], factor_min=0.8, factor_max=1.2, independent=False),
                RandomlyFlipAxis(apply_to=["lf", tgt], axis=-1),
                RandomlyFlipAxis(apply_to=["lf", tgt], axis=-2),
            )
            batch_preprocessing = ComposedTransform(
                RandomRotate90(apply_to=["lf", tgt]), ChannelFromLightField(apply_to={"lf": "lfc"}, nnum=nnum)
            )
        else:
            sample_preprocessing = ComposedTransform(
                Crop(apply_to=tgt, crop=((0, None), (35, -35), (shrink, -shrink), (shrink, -shrink))),
                Normalize01Dataset(apply_to="lf", min_percentile=5.0, max_percentile=99.8),
                Normalize01Dataset(apply_to=tgt, min_percentile=5.0, max_percentile=99.99),
                ChannelFromLightField(apply_to={"lf": "lfc"}, nnum=nnum),
            )
            batch_preprocessing = ComposedTransform()

        batch_preprocessing_in_step = Cast(apply_to=["lfc", tgt], dtype="float32", device="cuda", non_blocking=True)
        batch_postprocessing = ComposedTransform(
            Assert(apply_to="pred", expected_tensor_shape=(None, 1, z_out, None, None))
        )

    elif dataset_name in [DatasetChoice.beads_sample0, DatasetChoice.beads_highc_b]:
        tgt = "ls_reg"
        if dataset_name == DatasetChoice.beads_highc_b and dataset_part == DatasetPart.train:
            if scale != 8:
                # due to size the zenodo upload is resized to scale 8
                sample_precache_trf = [
                    {"Resize": {"apply_to": tgt, "shape": [1.0, 121, scale / 8, scale / 8], "order": 2}}
                ]
            else:
                sample_precache_trf = []
        else:
            sample_precache_trf = [
                {"Resize": {"apply_to": tgt, "shape": [1.0, 121, scale / 19, scale / 19], "order": 2}}
            ]

        sample_precache_trf.append({"Assert": {"apply_to": tgt, "expected_tensor_shape": [None, 1, 121, None, None]}})

        if dataset_part == DatasetPart.train:
            sample_preprocessing = ComposedTransform(
                Crop(apply_to=tgt, crop=((0, None), (35, -35), (shrink, -shrink), (shrink, -shrink))),
                Normalize01Dataset(apply_to="lf", min_percentile=5.0, max_percentile=99.8),
                Normalize01Dataset(apply_to=tgt, min_percentile=5.0, max_percentile=99.95),
                AdditiveGaussianNoise(apply_to="lf", sigma=0.1),
                AdditiveGaussianNoise(apply_to=tgt, sigma=0.05),
                RandomIntensityScale(apply_to=["lf", tgt], factor_min=0.8, factor_max=1.2, independent=False),
                RandomlyFlipAxis(apply_to=["lf", tgt], axis=-1),
                RandomlyFlipAxis(apply_to=["lf", tgt], axis=-2),
            )
            batch_preprocessing = ComposedTransform(
                RandomRotate90(apply_to=["lf", tgt]), ChannelFromLightField(apply_to={"lf": "lfc"}, nnum=nnum)
            )
        else:
            sample_preprocessing = ComposedTransform(
                Crop(apply_to=tgt, crop=((0, None), (35, -35), (shrink, -shrink), (shrink, -shrink))),
                Normalize01Dataset(apply_to="lf", min_percentile=5.0, max_percentile=99.8),
                Normalize01Dataset(apply_to=tgt, min_percentile=5.0, max_percentile=99.95),
                ChannelFromLightField(apply_to={"lf": "lfc"}, nnum=nnum),
            )
            batch_preprocessing = ComposedTransform()

        batch_preprocessing_in_step = Cast(apply_to=["lfc", tgt], dtype="float32", device="cuda", non_blocking=True)
        batch_postprocessing = ComposedTransform(
            Assert(apply_to="pred", expected_tensor_shape=(None, 1, z_out, None, None))
        )

    elif dataset_name in [
        DatasetChoice.heart_2020_02_fish1_static,
        DatasetChoice.heart_2020_02_fish1_static_sliced,
        DatasetChoice.heart_2020_02_fish2_static,
        DatasetChoice.heart_2020_02_fish2_static_sliced,
    ]:

        if "fish1" in dataset_name.name:
            crop_names.add("heart_2020_02_fish1_static")
        elif "fish2" in dataset_name.name:
            crop_names.add("heart_2020_02_fish2_static")
        else:
            raise NotImplementedError(dataset_name)

        spim_max_percentile = 99.8

        sample_precache_trf = []

        sample_preprocessing = ComposedTransform(
            CropWhatShrinkDoesNot(
                apply_to="lf", nnum=nnum, scale=scale, shrink=shrink, wrt_ref=True, crop_names=crop_names
            )
        )
        if load_lfd_and_care:
            sample_preprocessing += CropWhatShrinkDoesNot(
                apply_to="lfd", nnum=nnum, scale=scale, shrink=shrink, wrt_ref=False, crop_names=crop_names
            )

        if sliced or dynamic:
            tgt = "ls_slice"
            sample_preprocessing += CropLSforDynamicTraining(
                apply_to=tgt, crop_names=crop_names, nnum=nnum, scale=scale, z_ls_rescaled=z_ls_rescaled
            )
        else:
            tgt = "ls_trf"
            sample_preprocessing += CropWhatShrinkDoesNot(
                apply_to=tgt, nnum=nnum, scale=scale, shrink=shrink, wrt_ref=False, crop_names=crop_names
            )

            sample_preprocessing += Crop(
                apply_to=tgt, crop=((0, None), (0, None), (shrink, -shrink), (shrink, -shrink))
            )

        sample_preprocessing += Normalize01Dataset(apply_to="lf", min_percentile=5.0, max_percentile=99.8)
        sample_preprocessing += Normalize01Dataset(apply_to=tgt, min_percentile=5.0, max_percentile=spim_max_percentile)

        if dataset_part == DatasetPart.train:
            sample_preprocessing += ComposedTransform(
                RandomIntensityScale(apply_to=["lf", tgt], factor_min=0.8, factor_max=1.2, independent=False),
                PoissonNoise(apply_to="lf", peak=10),
                PoissonNoise(apply_to=tgt, peak=10),
                # AdditiveGaussianNoise(apply_to="lf", sigma=0.1),
                # AdditiveGaussianNoise(apply_to=spim, sigma=0.05),
                RandomlyFlipAxis(apply_to=["lf", tgt], axis=-1),
                RandomlyFlipAxis(apply_to=["lf", tgt], axis=-2),
            )
            batch_preprocessing = ComposedTransform(
                RandomRotate90(apply_to=["lf", tgt]), ChannelFromLightField(apply_to={"lf": "lfc"}, nnum=nnum)
            )
        else:
            sample_preprocessing += ComposedTransform(ChannelFromLightField(apply_to={"lf": "lfc"}, nnum=nnum))
            batch_preprocessing = ComposedTransform()

        batch_preprocessing_in_step = ComposedTransform(
            Cast(apply_to=["lfc", tgt], dtype="float32", device="cuda", non_blocking=True)
        )
        if load_lfd_and_care:
            batch_preprocessing_in_step += Cast(apply_to=["lfd"], dtype="float32", device="cuda", non_blocking=True)

        batch_postprocessing = ComposedTransform(
            Assert(apply_to="pred", expected_tensor_shape=(None, 1, z_out, None, None))
        )
        # sample_preprocessing += CropWhatShrinkDoesNot(
        #     apply_to=spim, nnum=nnum, scale=scale, shrink=shrink, wrt_ref=False, crop_names=crop_names
        # )
        #
        # sample_preprocessing += Crop(apply_to=spim, crop=((0, None), (0, None), (shrink, -shrink), (shrink, -shrink)))
        #
        # sample_preprocessing += Normalize01Dataset(apply_to="lf", min_percentile=5.0, max_percentile=99.8)
        # sample_preprocessing += Normalize01Dataset(
        #     apply_to=spim, min_percentile=5.0, max_percentile=spim_max_percentile
        # )
        #
        # if dataset_part == DatasetPart.train:
        #     raise NotImplementedError(dataset_part)
        # else:
        #     sample_preprocessing += ComposedTransform(ChannelFromLightField(apply_to={"lf": "lfc"}, nnum=nnum))
        #     batch_preprocessing = ComposedTransform()
        #
        # batch_preprocessing_in_step = Cast(apply_to=["lfc", spim], dtype="float32", device="cuda", non_blocking=True)
        # batch_postprocessing = ComposedTransform(
        #     Assert(apply_to="pred", expected_tensor_shape=(None, 1, z_out, None, None))
        # )

    elif dataset_name == DatasetChoice.heart_dyn_refine:
        # pred_z_min = 83.8
        # pred_z_max = 754.2
        crop_names.add("Heart_tightCrop")
        sample_precache_trf = []

        tgt = "ls_slice"
        spim_max_percentile = 99.8

        sample_preprocessing = ComposedTransform(
            CropWhatShrinkDoesNot(
                apply_to="lf", nnum=nnum, scale=scale, shrink=shrink, wrt_ref=True, crop_names=crop_names
            )
        )
        sample_preprocessing += CropLSforDynamicTraining(
            apply_to=tgt, crop_names=crop_names, nnum=nnum, scale=scale, z_ls_rescaled=z_ls_rescaled
        )

        sample_preprocessing += Normalize01Dataset(apply_to="lf", min_percentile=5.0, max_percentile=99.8)
        sample_preprocessing += Normalize01Dataset(apply_to=tgt, min_percentile=5.0, max_percentile=spim_max_percentile)

        if dataset_part == DatasetPart.train:
            sample_preprocessing += ComposedTransform(
                RandomIntensityScale(apply_to=["lf", tgt], factor_min=0.8, factor_max=1.2, independent=False),
                PoissonNoise(apply_to="lf", peak=10),
                PoissonNoise(apply_to=tgt, peak=10),
                # AdditiveGaussianNoise(apply_to="lf", sigma=0.1),
                # AdditiveGaussianNoise(apply_to=spim, sigma=0.05),
                RandomlyFlipAxis(apply_to=["lf", tgt], axis=-1),
                RandomlyFlipAxis(apply_to=["lf", tgt], axis=-2),
            )
            batch_preprocessing = ComposedTransform(
                RandomRotate90(apply_to=["lf", tgt]), ChannelFromLightField(apply_to={"lf": "lfc"}, nnum=nnum)
            )
        else:
            sample_preprocessing += ComposedTransform(ChannelFromLightField(apply_to={"lf": "lfc"}, nnum=nnum))
            batch_preprocessing = ComposedTransform()

        batch_preprocessing_in_step = Cast(apply_to=["lfc", tgt], dtype="float32", device="cuda", non_blocking=True)
        batch_postprocessing = ComposedTransform(
            Assert(apply_to="pred", expected_tensor_shape=(None, 1, z_out, None, None))
        )

    elif (
        dataset_name
        in [
            DatasetChoice.heart_static_sample0,
            DatasetChoice.heart_static_a,
            DatasetChoice.heart_static_b,
            DatasetChoice.heart_static_c,
        ]
        or dataset_part != DatasetPart.test
        and dataset_name
        in [
            DatasetChoice.heart_static_fish2,
            DatasetChoice.heart_static_fish2_sliced,
            DatasetChoice.heart_static_fish2_f4,
            DatasetChoice.heart_static_fish2_f4_sliced,
        ]
    ):
        tgt = "ls_slice" if sliced or dynamic else "ls_trf"
        crop_names.add("Heart_tightCrop")
        crop_names.add("staticHeartFOV")

        sample_precache_trf = []

        if dataset_name == DatasetChoice.heart_static_a:
            spim_max_percentile = 99.99
        else:
            spim_max_percentile = 99.8

        sample_preprocessing = ComposedTransform(
            CropWhatShrinkDoesNot(
                apply_to="lf", nnum=nnum, scale=scale, shrink=shrink, wrt_ref=True, crop_names=crop_names
            )
        )
        if sliced or dynamic:
            assert tgt == "ls_slice"
            sample_preprocessing += CropLSforDynamicTraining(
                apply_to=tgt, crop_names=crop_names, nnum=nnum, scale=scale, z_ls_rescaled=z_ls_rescaled
            )
        else:
            assert tgt != "ls_slice"
            sample_preprocessing += CropWhatShrinkDoesNot(
                apply_to=tgt, nnum=nnum, scale=scale, shrink=shrink, wrt_ref=False, crop_names=crop_names
            )

            sample_preprocessing += Crop(
                apply_to=tgt, crop=((0, None), (0, None), (shrink, -shrink), (shrink, -shrink))
            )

        sample_preprocessing += Normalize01Dataset(apply_to="lf", min_percentile=5.0, max_percentile=99.8)
        sample_preprocessing += Normalize01Dataset(apply_to=tgt, min_percentile=5.0, max_percentile=spim_max_percentile)

        if dataset_part == DatasetPart.train:
            sample_preprocessing += ComposedTransform(
                RandomIntensityScale(apply_to=["lf", tgt], factor_min=0.8, factor_max=1.2, independent=False),
                PoissonNoise(apply_to="lf", peak=10),
                PoissonNoise(apply_to=tgt, peak=10),
                # AdditiveGaussianNoise(apply_to="lf", sigma=0.1),
                # AdditiveGaussianNoise(apply_to=spim, sigma=0.05),
                RandomlyFlipAxis(apply_to=["lf", tgt], axis=-1),
                RandomlyFlipAxis(apply_to=["lf", tgt], axis=-2),
            )
            batch_preprocessing = ComposedTransform(
                RandomRotate90(apply_to=["lf", tgt]), ChannelFromLightField(apply_to={"lf": "lfc"}, nnum=nnum)
            )
        else:
            sample_preprocessing += ComposedTransform(ChannelFromLightField(apply_to={"lf": "lfc"}, nnum=nnum))
            batch_preprocessing = ComposedTransform()

        batch_preprocessing_in_step = Cast(apply_to=["lfc", tgt], dtype="float32", device="cuda", non_blocking=True)
        batch_postprocessing = ComposedTransform(
            Assert(apply_to="pred", expected_tensor_shape=(None, 1, z_out, None, None))
        )

    elif dataset_name == DatasetChoice.heart_static_c_care_complex:
        tgt = "ls_trf"
        sample_precache_trf = []

        sample_preprocessing = ComposedTransform(
            Normalize01Dataset(apply_to=tgt, min_percentile=5.0, max_percentile=99.8)
        )
        batch_preprocessing = ComposedTransform()
        batch_preprocessing_in_step = Cast(
            apply_to=["lfd", "care", tgt], dtype="float32", device="cuda", non_blocking=True
        )
        batch_postprocessing = ComposedTransform(
            Assert(apply_to="pred", expected_tensor_shape=(None, 1, z_out, None, None))
        )

    elif (
        dataset_name
        in [
            DatasetChoice.heart_static_fish2,
            DatasetChoice.heart_static_fish2_sliced,
            DatasetChoice.heart_static_fish2_f4,
            DatasetChoice.heart_static_fish2_f4_sliced,
            DatasetChoice.heart_static_fish5_sliced,
        ]
        and dataset_part == DatasetPart.test
    ):
        tgt = "spim"
        sample_precache_trf = []

        sample_preprocessing = ComposedTransform(
            Assert(apply_to="lf", expected_tensor_shape=(None, 1, None, None)),
            Assert(apply_to="lfd", expected_tensor_shape=(None, 1, z_out, None, None)),
            Assert(apply_to="care", expected_tensor_shape=(None, 1, z_out, None, None)),
            ChannelFromLightField(apply_to={"lf": "lfc"}, nnum=nnum),
            Assert(apply_to=tgt, expected_tensor_shape=(None, 1, z_out, None, None)),
            Assert(apply_to="lfd", expected_shape_like_tensor=tgt),
            Assert(apply_to="care", expected_shape_like_tensor=tgt),
        )

        batch_preprocessing = ComposedTransform()
        batch_preprocessing_in_step = ComposedTransform(
            Cast(apply_to=["lfc", "lfd", "care", tgt], dtype="float32", device="cuda", non_blocking=True)
        )
        batch_postprocessing = ComposedTransform(
            Assert(apply_to="pred", expected_tensor_shape=(None, 1, z_out, None, None))
        )

    elif dataset_name == DatasetChoice.heart_dyn_refine_lfd and dataset_part == DatasetPart.test:  # todo: test
        tgt = "ls_slice"
        sample_precache_trf = []

        sample_preprocessing = ComposedTransform(
            Assert(apply_to="lfd", expected_tensor_shape=(None, 1, z_out, None, None)),
            Assert(apply_to="care", expected_tensor_shape=(None, 1, z_out, None, None)),
            Assert(apply_to=tgt, expected_tensor_shape=(None, 1, z_out, None, None)),
            Assert(apply_to="lfd", expected_shape_like_tensor=tgt),
            Assert(apply_to="care", expected_shape_like_tensor=tgt),
        )

        batch_preprocessing = ComposedTransform()
        batch_preprocessing_in_step = ComposedTransform(
            Cast(apply_to=["lfd", "care", tgt], dtype="float32", device="cuda", non_blocking=True)
        )
        batch_postprocessing = ComposedTransform(
            Assert(apply_to="pred", expected_tensor_shape=(None, 1, z_out, None, None))
        )

    elif dataset_name == DatasetChoice.train_on_lfd:
        tgt = "lfd"
        sample_precache_trf = [
            {"Crop": {"apply_to": "lf", "crop": [[0, None], [38, -38], [38, -38]]}},
            {"Crop": {"apply_to": "lfd", "crop": [[0, None], [0, None], [38, -38], [38, -38]]}},
            {"Resize": {"apply_to": "lfd", "shape": [1.0, 1.0, scale / nnum, scale / nnum], "order": 2}},
            {"Assert": {"apply_to": "lfd", "expected_tensor_shape": (None, 1, 79, None, None)}},
        ]

        sample_preprocessing = ComposedTransform()
        sample_preprocessing += Crop(apply_to=tgt, crop=((0, None), (15, 64), (shrink, -shrink), (shrink, -shrink)))

        sample_preprocessing += Normalize01Dataset(apply_to="lf", min_percentile=5.0, max_percentile=99.8)
        sample_preprocessing += Normalize01Dataset(apply_to=tgt, min_percentile=5.0, max_percentile=99.8)

        if dataset_part == DatasetPart.train:
            sample_preprocessing += ComposedTransform(
                RandomIntensityScale(apply_to=["lf", tgt], factor_min=0.8, factor_max=1.2, independent=False),
                PoissonNoise(apply_to="lf", peak=10),
                PoissonNoise(apply_to=tgt, peak=10),
                # AdditiveGaussianNoise(apply_to="lf", sigma=0.1),
                # AdditiveGaussianNoise(apply_to=spim, sigma=0.05),
                RandomlyFlipAxis(apply_to=["lf", tgt], axis=-1),
                RandomlyFlipAxis(apply_to=["lf", tgt], axis=-2),
            )
            batch_preprocessing = ComposedTransform(
                RandomRotate90(apply_to=["lf", tgt]), ChannelFromLightField(apply_to={"lf": "lfc"}, nnum=nnum)
            )
        else:
            sample_preprocessing += ComposedTransform(ChannelFromLightField(apply_to={"lf": "lfc"}, nnum=nnum))
            batch_preprocessing = ComposedTransform()

        batch_preprocessing_in_step = Cast(apply_to=["lfc", tgt], dtype="float32", device="cuda", non_blocking=True)
        batch_postprocessing = ComposedTransform(
            Assert(apply_to="pred", expected_tensor_shape=(None, 1, z_out, None, None))
        )
    elif dataset_part == DatasetPart.predict:
        tgt = None
        sample_precache_trf = []

        sample_preprocessing = ComposedTransform(
            Assert(apply_to="lf", expected_tensor_shape=(None, 1, None, None)),
            Normalize01Dataset(apply_to="lf", min_percentile=5.0, max_percentile=99.8),
            ChannelFromLightField(apply_to={"lf": "lfc"}, nnum=nnum),
        )

        batch_preprocessing = ComposedTransform()
        batch_preprocessing_in_step = ComposedTransform(
            Cast(apply_to="lfc", dtype="float32", device="cuda", non_blocking=True)
        )
        batch_postprocessing = ComposedTransform(
            Assert(apply_to="pred", expected_tensor_shape=(None, 1, z_out, None, None))
        )

    else:
        raise NotImplementedError(dataset_name, dataset_part)

    meta = {
        "nnum": nnum,
        "z_out": z_out,
        "scale": scale,
        "interpolation_order": interpolation_order,
        "pred_z_min": pred_z_min,
        "pred_z_max": pred_z_max,
        "z_ls_rescaled": z_ls_rescaled,
        "crop_names": crop_names,
    }
    if (sliced or dynamic) and not dyn_precomputed:
        assert tgt == "ls_slice"

        def get_affine_trf_dyn(apply_to: str, targe_to_compare_to: Union[str, Tuple[int, str]]):
            return ComposedTransform(
                AffineTransformationDynamicTraining(
                    apply_to=apply_to,
                    target_to_compare_to=targe_to_compare_to,
                    crop_names=crop_names,
                    nnum=nnum,
                    scale=scale,
                    pred_z_min=pred_z_min,
                    pred_z_max=pred_z_max,
                    z_ls_rescaled=z_ls_rescaled,
                    padding_mode="border",
                    interpolation_order=interpolation_order,
                )
            )

        if incl_pred_vol:
            batch_postprocessing += Identity(apply_to={"pred": "pred_vol"})
            batch_postprocessing += get_affine_trf_dyn(
                "pred_vol", (z_out, tgt)
            )  # transform pred volume to ls orientation

        batch_postprocessing += get_affine_trf_dyn(
            "pred", tgt
        )  # transform pred and sample only the z_slice of ls_slice

    if tgt is not None:
        batch_postprocessing += Assert(apply_to="pred", expected_shape_like_tensor=tgt)

        batch_premetric_trf = ComposedTransform(NormalizeMSE(apply_to="pred", target_name=tgt, return_alpha_beta=True))
    else:
        batch_premetric_trf = ComposedTransform()

    return TransformsPipeline(
        sample_precache_trf=sample_precache_trf,
        sample_preprocessing=sample_preprocessing,
        batch_preprocessing=batch_preprocessing,
        batch_preprocessing_in_step=batch_preprocessing_in_step,
        batch_postprocessing=batch_postprocessing,
        batch_premetric_trf=batch_premetric_trf,
        meta=meta,
        tgt_name=tgt,
        spatial_dims=spatial_dims,
    )
