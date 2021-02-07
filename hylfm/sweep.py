import hylfm  # noqa

import sys

import wandb
from hylfm.hylfm_types import DatasetChoice

from hylfm.model import HyLFM_Net
from hylfm.train import train

default_kwargs = dict(
    dataset=DatasetChoice.beads_highc_a.value,
    model=dict(
        nnum=19,
        z_out=49,
        kernel2d=3,
        c00_2d=488,
        c01_2d=488,
        c02_2d=None,
        c03_2d=None,
        c04_2d=None,
        up0_2d=244,
        c10_2d=244,
        c11_2d=None,
        c12_2d=None,
        c13_2d=None,
        c14_2d=None,
        up1_2d=None,
        c20_2d=None,
        c21_2d=None,
        c22_2d=None,
        c23_2d=None,
        c24_2d=None,
        up2_2d=None,
        c30_2d=None,
        c31_2d=None,
        c32_2d=None,
        c33_2d=None,
        c34_2d=None,
        last_kernel2d=1,
        cin_3d=7,
        kernel3d=3,
        c00_3d=7,
        c01_3d=None,
        c02_3d=None,
        c03_3d=None,
        c04_3d=None,
        up0_3d=7,
        c10_3d=7,
        c11_3d=7,
        c12_3d=None,
        c13_3d=None,
        c14_3d=None,
        up1_3d=None,
        c20_3d=None,
        c21_3d=None,
        c22_3d=None,
        c23_3d=None,
        c24_3d=None,
        up2_3d=None,
        c30_3d=None,
        c31_3d=None,
        c32_3d=None,
        c33_3d=None,
        c34_3d=None,
        init_fn=HyLFM_Net.InitName.xavier_uniform,
        final_activation=None,
    )
)

resume = sys.argv[-1] == "--resume"
wandb.init(config=default_kwargs, resume=resume)
config = wandb.config


if __name__ == "__main__":
    train(**config)
