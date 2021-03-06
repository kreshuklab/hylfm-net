from pathlib import Path

from hylfm.hylfm_types import (
    CriterionChoice,
    DatasetChoice,
    LRSchedThresMode,
    LRSchedulerChoice,
    MetricChoice,
    OptimizerChoice,
    PeriodUnit,
)
from hylfm.model import HyLFM_Net
from hylfm.train import train


if __name__ == "__main__":
    train(
        dataset=DatasetChoice.beads_highc_b,
        batch_multiplier=2,
        batch_size=1,
        crit_apply_weight_above_threshold=False,
        crit_beta=1.0,
        crit_decay_weight_by=0.8,
        crit_decay_weight_every_unit=PeriodUnit.epoch,
        crit_decay_weight_every_value=1,
        crit_decay_weight_limit=1.0,
        crit_ms_ssim_weight=0.01,
        crit_threshold=0.5,
        crit_weight=0.001,
        criterion=CriterionChoice.WeightedSmoothL1,
        data_range=1.0,
        eval_batch_size=1,
        interpolation_order=2,
        lr_sched_factor=0.5,
        lr_sched_patience=10,
        lr_sched_thres=0.0001,
        lr_sched_thres_mode=LRSchedThresMode.abs,
        lr_scheduler=LRSchedulerChoice.ReduceLROnPlateau,
        max_epochs=10,
        model_weights=None,  # Path()
        opt_lr=3e-4,
        opt_momentum=0.0,
        opt_weight_decay=0.0,
        optimizer=OptimizerChoice.Adam,
        patience=5,
        score_metric=MetricChoice.MS_SSIM,
        seed=None,
        validate_every_unit=PeriodUnit.epoch,
        validate_every_value=1,
        win_sigma=1.5,
        win_size=11,
        # model
        nnum=19,
        z_out=51,
        kernel2d=3,
        c00_2d=976,
        c01_2d=976,
        c02_2d=0,
        c03_2d=0,
        c04_2d=0,
        up0_2d=488,
        c10_2d=488,
        c11_2d=0,
        c12_2d=0,
        c13_2d=0,
        c14_2d=0,
        up1_2d=244,
        c20_2d=244,
        c21_2d=0,
        c22_2d=0,
        c23_2d=0,
        c24_2d=0,
        up2_2d=0,
        c30_2d=0,
        c31_2d=0,
        c32_2d=0,
        c33_2d=0,
        c34_2d=0,
        last_kernel2d=1,
        cin_3d=7,
        kernel3d=3,
        c00_3d=7,
        c01_3d=0,
        c02_3d=0,
        c03_3d=0,
        c04_3d=0,
        up0_3d=7,
        c10_3d=7,
        c11_3d=7,
        c12_3d=0,
        c13_3d=0,
        c14_3d=0,
        up1_3d=0,
        c20_3d=0,
        c21_3d=0,
        c22_3d=0,
        c23_3d=0,
        c24_3d=0,
        up2_3d=0,
        c30_3d=0,
        c31_3d=0,
        c32_3d=0,
        c33_3d=0,
        c34_3d=0,
        init_fn=HyLFM_Net.InitName.xavier_uniform_,
        final_activation=None,
    )
