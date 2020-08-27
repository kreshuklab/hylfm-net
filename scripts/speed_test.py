import math
from timeit import Timer

import torch
from hylfm.models.a04 import A04

# from collections import OrderedDict
# from torch.utils.data import DataLoader
# from hylfm.datasets import ZipDataset, collate_fn, get_dataset_from_info, get_tensor_info
# from hylfm.transformations import Cast, ChannelFromLightField, CropWhatShrinkDoesNot, Normalize01

if __name__ == "__main__":
    assert torch.cuda.device_count() == 1, torch.cuda.device_count()
    precision = "float"
    crop_name = "beads"  # Heart_tightCrop, gcamp, beads
    meta = {
        "z_out": 51 if crop_name == "beads" else 49,
        "nnum": 19,
        "interpolation_order": 2,
        "scale": 8,
        "shrink": 8,
        "crop_names": [crop_name],
        "z_ls_rescaled": 241,
        "pred_z_min": 142 if crop_name == "gcamp" else 0,
        "pred_z_max": 620 if crop_name == "gcamp" else 838,
    }
    # ds = ZipDataset(
    #     OrderedDict(
    #         [
    #             (
    #                 "lf",
    #                 get_dataset_from_info(
    #                     get_tensor_info("heart_dynamic.plane_120/2019-12-09_05.53.55", "lf", meta), cache=True
    #                 ),
    #             )
    #         ]
    #     )
    # )
    lower_bound, upper_bound = 0, 1
    while upper_bound > lower_bound:
        batch_size = math.ceil((upper_bound + lower_bound) / 2)
        print("test batch size", batch_size)
        try:
            # dl = DataLoader(ds, batch_size=batch_size, num_workers=0, collate_fn=collate_fn)
            # sample = next(iter(dl))
            # print("raw lf size", sample["lf"].shape)
            # assert sample["lf"].shape[0] == batch_size, sample["lf"].shape[0]
            # # print(sample["lf"].shape)
            # sample = CropWhatShrinkDoesNot(apply_to="lf", meta=meta, wrt_ref=True).apply(sample)
            # # print(sample["lf"].shape)
            # sample = Normalize01(apply_to="lf", min_percentile=5.0, max_percentile=99.8).apply(sample)
            # # print(sample["lf"].shape)
            # sample = ChannelFromLightField(apply_to="lf", nnum=meta["nnum"]).apply(sample)
            # # print(sample["lf"].shape)
            # sample = Cast(apply_to="lf", dtype=precision, device="cuda", non_blocking=False).apply(sample)
            #
            # print("prepared lfc", sample["lf"].shape, sample["lf"].dtype)
            # print('meta', sample["meta"])

            _lf_size = [batch_size]
            if crop_name == "Heart_tightCrop":
                _lf_size += [361, 65, 75]
            elif crop_name == "gcamp":
                _lf_size += [361, 68, 83]
            elif crop_name == "beads":
                _lf_size += [361, 49, 74]
            else:
                raise NotImplementedError(crop_name)

            lf_size = tuple(_lf_size)
            assert crop_name == "beads"  # nres2d, nred3d
            model = A04(
                input_name="lf",
                prediction_name="pred",
                # n_res2d=[488, 488, "u", 244, 244],
                # n_res3d=[[7, 7], [7], [7]],
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
                    model(
                        {
                            "lf": torch.empty(lf_size, dtype=torch.float32, device="cuda"),
                            "meta": [{"lf": {}}] * batch_size,
                        }
                    )
                    # model(sample)

            timer = Timer(forward)
            n = 1000
            total = timer.repeat(3, number=n)
            per_batch = [t / n for t in total]
            per_sample = [pb / batch_size for pb in per_batch]
            in_hz = [1 / ps for ps in per_sample]
            for tot, pb, ps, hz in zip(total, per_batch, per_sample, in_hz):
                print(
                    f"total: {tot}, batch_size: {batch_size:2}\n\tper batch:  {pb:5.5f} s\n\tper sample: {ps:5.5f} s\n\t        hz: {hz:3.3f}"
                )
        except RuntimeError as e:
            print(e)
            upper_bound = batch_size - 1
            print('upper bound', upper_bound)
        else:
            lower_bound = batch_size
            print('lower bound', lower_bound)
