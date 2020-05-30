from collections import OrderedDict
from collections import OrderedDict
from timeit import Timer

import torch
from torch.utils.data import DataLoader

from lnet.datasets import ZipDataset, get_collate_fn, get_dataset_from_info, get_tensor_info
from lnet.models.a04 import A04
from lnet.transformations import Cast, ChannelFromLightField, CropWhatShrinkDoesNot, Normalize01

if __name__ == "__main__":
    assert torch.cuda.device_count() == 1, torch.cuda.device_count()
    precision = "float"
    meta = {
        "z_out": 49,
        "nnum": 19,
        "interpolation_order": 2,
        "scale": 2,
        "shrink": 6,
        "crop_names": ["Heart_tightCrop"],
        "z_ls_rescaled": 241,
        "pred_z_min": 0,
        "pred_z_max": 838,
    }
    ds = ZipDataset(
        OrderedDict(
            [("lf", get_dataset_from_info(get_tensor_info("heart_dynamic.plane_120/2019-12-09_05.53.55", "lf", meta), cache=True))]
        )
    )
    for batch_size in [64, 128]:
        dl = DataLoader(ds, batch_size=batch_size, num_workers=0, collate_fn=get_collate_fn(lambda x: x))
        sample = next(iter(dl))
        assert sample["lf"].shape[0] == batch_size, sample["lf"].shape[0]
        # print(sample["lf"].shape)
        sample = CropWhatShrinkDoesNot(apply_to="lf", meta=meta, wrt_ref=True).apply(sample)
        # print(sample["lf"].shape)
        sample = Normalize01(apply_to="lf", min_percentile=5.0, max_percentile=99.8).apply(sample)
        # print(sample["lf"].shape)
        sample = ChannelFromLightField(apply_to="lf", nnum=meta["nnum"]).apply(sample)
        # print(sample["lf"].shape)
        sample = Cast(apply_to="lf", dtype=precision, device="cuda", non_blocking=False).apply(sample)

        model = A04(
            input_name="lf",
            prediction_name="pred",
            n_res2d=[488, 488, "u", 244, 244],
            n_res3d=[[7], [7], [7]],
            z_out=meta["z_out"],
            nnum=meta["nnum"],
            scale=meta["scale"],
            shrink=meta["shrink"],
        )
        model = model.to("cuda")
        model.eval()
        # out = model(sample)
        # print(out["pred"].shape)

        def forward():
            with torch.no_grad():
                model(sample)

        forward()
        timer = Timer(forward)
        n = 100
        per_batch = [t / n for t in timer.repeat(5, number=n)]
        per_sample = [pb / batch_size for pb in per_batch]
        in_hz = [1 / ps for ps in per_sample]
        for pb, ps, hz in zip(per_batch, per_sample, in_hz):
            print(f"batch_size: {batch_size:2}\n\tper batch:  {pb:5.5f} s\n\tper sample: {ps:5.5f} s\n\t        hz: {hz:3.3f}")

        print()
