import logging.config
from pathlib import Path
from typing import List, Optional, Tuple, Union

from merge_args_manual import merge_args

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import torch
import typer

from hylfm.model import HyLFM_Net

CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s %(name)s %(levelname)s %(message)s",  # .%(msecs)03d [%(processName)s/%(threadName)s]
            "datefmt": "%H:%M:%S",
        }
    },
    "handlers": {
        "default": {
            "level": "DEBUG",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
            "formatter": "default",
        }
    },
    "loggers": {
        "": {"handlers": ["default"], "level": "INFO", "propagate": True},
        "lensletnet.datasets": {"handlers": ["default"], "level": "INFO", "propagate": False},
        "lensletnet": {"handlers": ["default"], "level": "INFO", "propagate": False},
    },
}
logging.config.dictConfig(CONFIG)
logger = logging.getLogger(__name__)

app = typer.Typer()


@app.command("model")
def get_model(
    # *,  # todo: figure out why key word only arguments don't seem not work with typer/merge_args!?
    nnum: int = 19,
    z_out: int = 49,
    kernel2d: int = 3,
    # note: typer does (not yet) support Ellipsis, e.g. Tuple[int, ...], but returns tuple when List is specified
    c00_2d: Optional[int] = 488,
    c01_2d: Optional[int] = 488,
    c02_2d: Optional[int] = None,
    c03_2d: Optional[int] = None,
    c04_2d: Optional[int] = None,
    up0_2d: Optional[int] = 244,
    c10_2d: Optional[int] = 244,
    c11_2d: Optional[int] = None,
    c12_2d: Optional[int] = None,
    c13_2d: Optional[int] = None,
    c14_2d: Optional[int] = None,
    up1_2d: Optional[int] = None,
    c20_2d: Optional[int] = None,
    c21_2d: Optional[int] = None,
    c22_2d: Optional[int] = None,
    c23_2d: Optional[int] = None,
    c24_2d: Optional[int] = None,
    up2_2d: Optional[int] = None,
    c30_2d: Optional[int] = None,
    c31_2d: Optional[int] = None,
    c32_2d: Optional[int] = None,
    c33_2d: Optional[int] = None,
    c34_2d: Optional[int] = None,
    last_kernel2d: int = 1,
    cin_3d: int = 7,
    kernel3d: int = 3,
    c00_3d: Optional[int] = 7,
    c01_3d: Optional[int] = None,
    c02_3d: Optional[int] = None,
    c03_3d: Optional[int] = None,
    c04_3d: Optional[int] = None,
    up0_3d: Optional[int] = 7,
    c10_3d: Optional[int] = 7,
    c11_3d: Optional[int] = 7,
    c12_3d: Optional[int] = None,
    c13_3d: Optional[int] = None,
    c14_3d: Optional[int] = None,
    up1_3d: Optional[int] = None,
    c20_3d: Optional[int] = None,
    c21_3d: Optional[int] = None,
    c22_3d: Optional[int] = None,
    c23_3d: Optional[int] = None,
    c24_3d: Optional[int] = None,
    up2_3d: Optional[int] = None,
    c30_3d: Optional[int] = None,
    c31_3d: Optional[int] = None,
    c32_3d: Optional[int] = None,
    c33_3d: Optional[int] = None,
    c34_3d: Optional[int] = None,
    init_fn: HyLFM_Net.InitName = HyLFM_Net.InitName.xavier_uniform,
    final_activation: Optional[str] = None,
    checkpoint: Optional[Path] = Path(
        r"C:\Users\fbeut\Desktop\hylfm_stuff\old_checkpoints\v1_checkpoint_498_MS_SSIM=0.9710696664723483.pth"
    ),
):
    c_res2d = [
        c00_2d,
        c01_2d,
        c02_2d,
        c03_2d,
        c04_2d,
        None if up0_2d is None else f"u{up0_2d}",
        c10_2d,
        c11_2d,
        c12_2d,
        c13_2d,
        c14_2d,
        None if up1_2d is None else f"u{up1_2d}",
        c20_2d,
        c21_2d,
        c22_2d,
        c23_2d,
        c24_2d,
        None if up2_2d is None else f"u{up2_2d}",
        c30_2d,
        c31_2d,
        c32_2d,
        c33_2d,
        c34_2d,
    ]
    c_res3d = [
        c00_3d,
        c01_3d,
        c02_3d,
        c03_3d,
        c04_3d,
        None if up0_3d is None else f"u{up0_3d}",
        c10_3d,
        c11_3d,
        c12_3d,
        c13_3d,
        c14_3d,
        None if up1_3d is None else f"u{up1_3d}",
        c20_3d,
        c21_3d,
        c22_3d,
        c23_3d,
        c24_3d,
        None if up2_3d is None else f"u{up2_3d}",
        c30_3d,
        c31_3d,
        c32_3d,
        c33_3d,
        c34_3d,
    ]
    if torch.cuda.device_count() > 1:
        raise RuntimeError(f"Set CUDA_VISIBLE_DEVICES!")

    model = HyLFM_Net(
        z_out=z_out,
        nnum=nnum,
        kernel2d=kernel2d,
        c_res2d=[c for c in c_res2d if c is not None],
        last_kernel2d=last_kernel2d,
        c_in_3d=cin_3d,
        kernel3d=kernel3d,
        c_res3d=[c for c in c_res3d if c is not None],
        init_fn=init_fn,
        final_activation=final_activation,
    )

    if checkpoint is not None:
        # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        device = torch.device("cpu")
        state = torch.load(str(checkpoint), map_location=device)["model"]
        model.load_state_dict(state, strict=True)

    return model


@app.command()
@merge_args(get_model)
def train(**model_kwargs,):
    print(model_kwargs)
    model = get_model(**model_kwargs)
    # if checkpoint == "small_beads_demo":
    #     small_beads_demo_doi = "10.5281/zenodo.4036556"
    #     small_beads_demo_file_name = "small_beads_v1_weights_SmoothL1Loss%3D-0.00012947025970788673.pth"
    #     checkpoint = settings.download_dir / small_beads_demo_doi / small_beads_demo_file_name
    #     download_file_from_zenodo(small_beads_demo_doi, small_beads_demo_file_name, checkpoint)
    # elif checkpoint is not None:
    #     checkpoint = Path(checkpoint)

    # from hylfm.setup import Setup
    #
    # setup = Setup.from_yaml(experiment_config)
    #
    # if args.setup:
    #     log_path = setup.setup()
    #     shutil.rmtree(log_path)
    # else:
    #     logger.info(
    #         "run tensorboard with:\ntensorboard --logdir=%s\nor:\ntensorboard --logdir=%s",
    #         settings.log_dir,
    #         setup.log_path.parent,
    #     )
    #     log_path = setup.run()
    #     logger.info("done logging to %s", log_path)


@app.command()
def test():
    pass


if __name__ == "__main__":
    app()
