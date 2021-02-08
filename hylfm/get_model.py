from typing import Optional

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import torch
import typer

from hylfm.model import HyLFM_Net

app = typer.Typer()


@app.command("model")
def get_model(
    # *,  # todo: figure out why key word only arguments don't seem not work with typer/merge_args!?
    nnum: int = 19,
    z_out: int = typer.Option(49, "--z_out"),
    kernel2d: int = 3,
    # note: typer does (not yet) support Ellipsis, e.g. Tuple[int, ...], but returns tuple when List is specified
    c00_2d: Optional[int] = typer.Option(488, "--c00_2d"),
    c01_2d: Optional[int] = typer.Option(488, "--c01_2d"),
    c02_2d: Optional[int] = typer.Option(0, "--c02_2d"),
    c03_2d: Optional[int] = typer.Option(0, "--c03_2d"),
    c04_2d: Optional[int] = typer.Option(0, "--c04_2d"),
    up0_2d: Optional[int] = typer.Option(244, "--up0_2d"),
    c10_2d: Optional[int] = typer.Option(244, "--c10_2d"),
    c11_2d: Optional[int] = typer.Option(0, "--c11_2d"),
    c12_2d: Optional[int] = typer.Option(0, "--c12_2d"),
    c13_2d: Optional[int] = typer.Option(0, "--c13_2d"),
    c14_2d: Optional[int] = typer.Option(0, "--c14_2d"),
    up1_2d: Optional[int] = typer.Option(0, "--up1_2d"),
    c20_2d: Optional[int] = typer.Option(0, "--c20_2d"),
    c21_2d: Optional[int] = typer.Option(0, "--c21_2d"),
    c22_2d: Optional[int] = typer.Option(0, "--c22_2d"),
    c23_2d: Optional[int] = typer.Option(0, "--c23_2d"),
    c24_2d: Optional[int] = typer.Option(0, "--c24_2d"),
    up2_2d: Optional[int] = typer.Option(0, "--up2_2d"),
    c30_2d: Optional[int] = typer.Option(0, "--c30_2d"),
    c31_2d: Optional[int] = typer.Option(0, "--c31_2d"),
    c32_2d: Optional[int] = typer.Option(0, "--c32_2d"),
    c33_2d: Optional[int] = typer.Option(0, "--c33_2d"),
    c34_2d: Optional[int] = typer.Option(0, "--c34_2d"),
    last_kernel2d: int = typer.Option(1, "--last_kernel2d"),
    cin_3d: int = typer.Option(7, "--cin_3d"),
    kernel3d: int = 3,
    c00_3d: Optional[int] = typer.Option(7, "--c00_3d"),
    c01_3d: Optional[int] = typer.Option(0, "--c01_3d"),
    c02_3d: Optional[int] = typer.Option(0, "--c02_3d"),
    c03_3d: Optional[int] = typer.Option(0, "--c03_3d"),
    c04_3d: Optional[int] = typer.Option(0, "--c04_3d"),
    up0_3d: Optional[int] = typer.Option(7, "--up0_3d"),
    c10_3d: Optional[int] = typer.Option(7, "--c10_3d"),
    c11_3d: Optional[int] = typer.Option(7, "--c11_3d"),
    c12_3d: Optional[int] = typer.Option(0, "--c12_3d"),
    c13_3d: Optional[int] = typer.Option(0, "--c13_3d"),
    c14_3d: Optional[int] = typer.Option(0, "--c14_3d"),
    up1_3d: Optional[int] = typer.Option(0, "--up1_3d"),
    c20_3d: Optional[int] = typer.Option(0, "--c20_3d"),
    c21_3d: Optional[int] = typer.Option(0, "--c21_3d"),
    c22_3d: Optional[int] = typer.Option(0, "--c22_3d"),
    c23_3d: Optional[int] = typer.Option(0, "--c23_3d"),
    c24_3d: Optional[int] = typer.Option(0, "--c24_3d"),
    up2_3d: Optional[int] = typer.Option(0, "--up2_3d"),
    c30_3d: Optional[int] = typer.Option(0, "--c30_3d"),
    c31_3d: Optional[int] = typer.Option(0, "--c31_3d"),
    c32_3d: Optional[int] = typer.Option(0, "--c32_3d"),
    c33_3d: Optional[int] = typer.Option(0, "--c33_3d"),
    c34_3d: Optional[int] = typer.Option(0, "--c34_3d"),
    init_fn: HyLFM_Net.InitName = typer.Option(HyLFM_Net.InitName.xavier_uniform_, "--init_fn"),
    final_activation: Optional[str] = typer.Option(None, "--final_activation"),
):
    c_res2d = [
        c00_2d,
        c01_2d,
        c02_2d,
        c03_2d,
        c04_2d,
        f"u{up0_2d}" if up0_2d else 0,
        c10_2d,
        c11_2d,
        c12_2d,
        c13_2d,
        c14_2d,
        f"u{up1_2d}" if up1_2d else 0,
        c20_2d,
        c21_2d,
        c22_2d,
        c23_2d,
        c24_2d,
        f"u{up2_2d}" if up2_2d else 0,
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
        f"u{up0_3d}" if up0_3d else 0,
        c10_3d,
        c11_3d,
        c12_3d,
        c13_3d,
        c14_3d,
        f"u{up1_3d}" if up1_3d else 0,
        c20_3d,
        c21_3d,
        c22_3d,
        c23_3d,
        c24_3d,
        f"u{up2_3d}" if up2_3d else 0,
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
        c_res2d=[c for c in c_res2d if c],
        last_kernel2d=last_kernel2d,
        c_in_3d=cin_3d,
        kernel3d=kernel3d,
        c_res3d=[c for c in c_res3d if c],
        init_fn=init_fn,
        final_activation=final_activation,
    )
    model = model.cuda()
    return model
