from pathlib import Path

import torch

from lnet.models.a01 import A01
from lnet.models.a04 import A04


def test_for_hear():
    assert torch.cuda.device_count() == 1
    device = torch.device("cuda")
    checkpoint_path = Path("/g/kreshuk/beuttenm/repos/llnet/logs/fish/fdyn1_a02/20-02-24_09-12-36/checkpoint.pth")
    new_path = Path(
        "/g/kreshuk/beuttenm/repos/lnet/ref_data/mapped_checkpoints/fish/fdyn1_a02/20-02-24_09-12-36/checkpoint.pth"
    )
    # checkpoint_path = Path("/g/kreshuk/beuttenm/repos/llnet/logs/fish/fdyn1_a02/20-02-21_08-54-55/models/v0_model_6.pth")
    # new_path = Path("/g/kreshuk/beuttenm/repos/lnet/ref_data/mapped_checkpoints/fish/fdyn1_a02/220-02-21_08-54-55/models/v0_model_6.pth")
    # checkpoint_path = Path("/g/kreshuk/beuttenm/repos/llnet/logs/fish/fdyn1_a01/20-02-22_09-22-48/models/v0_model_13.pth")
    # new_path = Path("/g/kreshuk/beuttenm/repos/lnet/ref_data/mapped_checkpoints/fish/fdyn1_a01/220-02-22_09-22-48/models/v0_model_13.pth")

    new_path.parent.mkdir(parents=True, exist_ok=True)
    assert checkpoint_path.exists()
    weights = torch.load(checkpoint_path, map_location=device)
    torch.save({"model": weights}, new_path)

    # 1,2
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
    # 3
    # model = A01(
    #     input_name="lfc",
    #     prediction_name="pred",
    #     z_out=49,
    #     nnum=19,
    #     n_res2d=[128, 64],
    #     inplanes_3d=32,
    #     n_res3d=[[32, 16], [8, 4]],
    #     # last_2d_kernel=(3, 3),
    # )
    model = model.to(device=device)
    model.load_state_dict(weights, strict=True)
    ipt = torch.ones(1, 19 ** 2, 1273 // 19, 1463 // 19, device=device)
    print("ipt", ipt.shape)
    out = model({"lfc": ipt})["pred"]
    print("out", out.shape)
    print("srhink", model.get_shrinkage())
    print("scale", model.get_scaling())
