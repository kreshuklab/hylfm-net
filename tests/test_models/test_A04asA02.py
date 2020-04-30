from pathlib import Path

import torch

from lnet.models.a04 import A04


def test_for_hear():
    assert torch.cuda.device_count() == 1
    device = torch.device("cuda")
    checkpoint_path = Path("/g/kreshuk/beuttenm/repos/llnet/logs/fish/fdyn1_a02/20-02-24_09-12-36/checkpoint.pth")
    assert checkpoint_path.exists()
    weights = torch.load(checkpoint_path, map_location=device)
    model = A04(
        input_name="lfc",
        prediction_name="pred",
        z_out=49,
        nnum=19,
        n_res2d=[128, 128, "u", 64, 64],
        inplanes_3d=32,
        n_res3d=[[32, 8], [8], [1]],
        last_2d_kernel=(3, 3),
    )
    model = model.to(device=device)
    model.load_state_dict(weights, strict=True)
