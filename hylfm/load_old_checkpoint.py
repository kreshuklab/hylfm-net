from pathlib import Path

from hylfm.checkpoint import TrainRunConfig
from hylfm.hylfm_types import DatasetChoice


def get_config_for_old_checkpoint(checkpoint: Path) -> TrainRunConfig:
    dataset = None
    if checkpoint.name in [
        "v1_checkpoint_498_MS_SSIM=0.9710696664723483.pth",  # heart stat
        "v1_checkpoint_MSSSIM=0.6722144321961836.pth",
        "v1_checkpoint_27500_ms_ssim-scaled=0.8430055000565269.pth",
        "v1_checkpoint_37400_ms_ssim-scaled=0.8433429219506003.pth",  # repos/hylfm-net/logs/train/heart/fake_dyn/from_scratch/21-01-12_21-05-48
        "v1_checkpoint_28200_ms_ssim-scaled=0.8401095325296576.pth",
        "v1_checkpoint_37500_ms_ssim-scaled=0.8358002250844782.pth",
        "v1_checkpoint_0_ms_ssim-scaled=0.8727825609120455.pth",
        "v1_checkpoint_9900_ms_ssim-scaled=0.8582265810533003.pth",
        "v1_checkpoint_6600_ms_ssim-scaled=0.9658018271759073.pth",  # refined old stat on fish2 dyn refine
    ]:
        model_config = {
            "nnum": 19,
            "z_out": 49,
            "kernel2d": 3,
            "c00_2d": 488,
            "c01_2d": 488,
            "c02_2d": None,
            "c03_2d": None,
            "c04_2d": None,
            "up0_2d": 244,
            "c10_2d": 244,
            "c11_2d": None,
            "c12_2d": None,
            "c13_2d": None,
            "c14_2d": None,
            "up1_2d": None,
            "c20_2d": None,
            "c21_2d": None,
            "c22_2d": None,
            "c23_2d": None,
            "c24_2d": None,
            "up2_2d": None,
            "c30_2d": None,
            "c31_2d": None,
            "c32_2d": None,
            "c33_2d": None,
            "c34_2d": None,
            "last_kernel2d": 1,
            "cin_3d": 7,
            "kernel3d": 3,
            "c00_3d": 7,
            "c01_3d": None,
            "c02_3d": None,
            "c03_3d": None,
            "c04_3d": None,
            "up0_3d": 7,
            "c10_3d": 7,
            "c11_3d": 7,
            "c12_3d": None,
            "c13_3d": None,
            "c14_3d": None,
            "up1_3d": None,
            "c20_3d": None,
            "c21_3d": None,
            "c22_3d": None,
            "c23_3d": None,
            "c24_3d": None,
            "up2_3d": None,
            "c30_3d": None,
            "c31_3d": None,
            "c32_3d": None,
            "c33_3d": None,
            "c34_3d": None,
            "init_fn": "xavier_uniform",
            "final_activation": None,
        }
    elif checkpoint.name in [
        "v1_checkpoint_SmoothL1Loss=-0.00012947025970788673.pth",
        "small_beads_v1_weights_SmoothL1Loss%3D-0.00012947025970788673.pth",
    ]:  # beads f8
        dataset = DatasetChoice.beads_highc_a
        model_config = {
            "nnum": 19,
            "z_out": 51,
            "kernel2d": 3,
            "c00_2d": 976,
            "c01_2d": 976,
            "c02_2d": None,
            "c03_2d": None,
            "c04_2d": None,
            "up0_2d": 488,
            "c10_2d": 488,
            "c11_2d": None,
            "c12_2d": None,
            "c13_2d": None,
            "c14_2d": None,
            "up1_2d": 244,
            "c20_2d": 244,
            "c21_2d": None,
            "c22_2d": None,
            "c23_2d": None,
            "c24_2d": None,
            "up2_2d": None,
            "c30_2d": None,
            "c31_2d": None,
            "c32_2d": None,
            "c33_2d": None,
            "c34_2d": None,
            "last_kernel2d": 1,
            "cin_3d": 7,
            "kernel3d": 3,
            "c00_3d": 7,
            "c01_3d": None,
            "c02_3d": None,
            "c03_3d": None,
            "c04_3d": None,
            "up0_3d": 7,
            "c10_3d": 7,
            "c11_3d": 7,
            "c12_3d": None,
            "c13_3d": None,
            "c14_3d": None,
            "up1_3d": None,
            "c20_3d": None,
            "c21_3d": None,
            "c22_3d": None,
            "c23_3d": None,
            "c24_3d": None,
            "up2_3d": None,
            "c30_3d": None,
            "c31_3d": None,
            "c32_3d": None,
            "c33_3d": None,
            "c34_3d": None,
            "init_fn": "xavier_uniform",
            "final_activation": None,
        }

    elif checkpoint.name == "v1_checkpoint_SmoothL1Loss=-0.00016112386947497725.pth":  # beads f4
        model_config = {
            "nnum": 19,
            "z_out": 51,
            "kernel2d": 3,
            "c00_2d": 488,
            "c01_2d": 488,
            "c02_2d": None,
            "c03_2d": None,
            "c04_2d": None,
            "up0_2d": 244,
            "c10_2d": 244,
            "c11_2d": None,
            "c12_2d": None,
            "c13_2d": None,
            "c14_2d": None,
            "up1_2d": None,
            "c20_2d": None,
            "c21_2d": None,
            "c22_2d": None,
            "c23_2d": None,
            "c24_2d": None,
            "up2_2d": None,
            "c30_2d": None,
            "c31_2d": None,
            "c32_2d": None,
            "c33_2d": None,
            "c34_2d": None,
            "last_kernel2d": 1,
            "cin_3d": 7,
            "kernel3d": 3,
            "c00_3d": 7,
            "c01_3d": None,
            "c02_3d": None,
            "c03_3d": None,
            "c04_3d": None,
            "up0_3d": 7,
            "c10_3d": 7,
            "c11_3d": 7,
            "c12_3d": None,
            "c13_3d": None,
            "c14_3d": None,
            "up1_3d": None,
            "c20_3d": None,
            "c21_3d": None,
            "c22_3d": None,
            "c23_3d": None,
            "c24_3d": None,
            "up2_3d": None,
            "c30_3d": None,
            "c31_3d": None,
            "c32_3d": None,
            "c33_3d": None,
            "c34_3d": None,
            "init_fn": "xavier_uniform",
            "final_activation": None,
        }
    else:
        raise NotImplementedError(checkpoint)

    return TrainRunConfig(
        batch_multiplier=1,
        batch_size=1,
        crit_apply_weight_above_threshold=None,
        crit_beta=None,
        crit_decay_weight_by=None,
        crit_decay_weight_every_unit=None,
        crit_decay_weight_every_value=None,
        crit_decay_weight_limit=None,
        crit_ms_ssim_weight=None,
        crit_threshold=None,
        crit_weight=None,
        criterion=None,
        data_range=1.0,
        dataset=dataset,
        eval_batch_size=1,
        interpolation_order=2,
        lr_sched_factor=None,
        lr_sched_patience=None,
        lr_sched_thres=None,
        lr_sched_thres_mode=None,
        lr_scheduler=None,
        max_epochs=None,
        model=model_config,
        model_weights=checkpoint,
        opt_lr=None,
        opt_momentum=None,
        opt_weight_decay=None,
        optimizer=None,
        patience=None,
        save_output_to_disk=None,
        score_metric=None,
        seed=None,
        validate_every_unit=None,
        validate_every_value=None,
        win_sigma=1.5,
        win_size=11,
        hylfm_version="0.0.0",
        point_cloud_threshold=1.0,
    )
