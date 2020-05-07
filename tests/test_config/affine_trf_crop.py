from collections import OrderedDict

from lnet.datasets import ZipDataset, get_dataset_from_info, get_tensor_info
from lnet.datasets.heart_utils import get_lf_shape, get_ls_shape
from lnet.transformations.affine_utils import get_ref_crop_out
from lnet.transformations.utils import get_composed_transformation_from_config


def test_crop():
    crop_name = "Heart_tightCrop"
    if crop_name == "Heart_tightCrop":
        info_name = "static_heart.2019-12-08_06.35.52"
    elif crop_name == "staticHeartFOV":
        info_name = "static_heart.2019-12-08_06.57.57"
    else:
        raise NotImplementedError

    meta = {
        "z_out": 48,
        "nnum": 19,
        "interpolation_order": 2,
        "scale": 2,
        "shrink": 6,
        "ls_z_min": 0,
        "ls_z_max": 241,
        "pred_z_min": 0,
        "pred_z_max": 838,
    }
    assert meta["ls_z_max"] > 0

    # f2:
    # Heart_tightCrop ref_crop_out: auto: [(18, -157), (139, -111), (145, -119)]

    target_to_compare_to = "ls"
    # [meta["z_out"]] + [round(xy / meta["nnum"] * meta["scale"]) for xy in get_lf_shape(crop_name)]

    ref_crop_in = (
        (meta["pred_z_min"], meta["pred_z_max"]),
        (round(meta["shrink"] * meta["nnum"] / meta["scale"]), -round(meta["shrink"] * meta["nnum"] / meta["scale"])),
        (round(meta["shrink"] * meta["nnum"] / meta["scale"]), -round(meta["shrink"] * meta["nnum"] / meta["scale"])),
    )
    ref_crop_out = get_ref_crop_out(crop_name, ref_crop_in, inverted=False)

    trf_config = [
        {
            "Assert": {
                "apply_to": target_to_compare_to,
                "expected_tensor_shape": [None, 1, None]
                + [round(s / meta["nnum"] * meta["scale"]) for s in get_ls_shape(crop_name)[1:]],
            }
        },
        {
            "Crop": {
                "apply_to": target_to_compare_to,
                "crop": [(0, None), [c * meta["z_out"] / get_ls_shape(crop_name)[0] for c in ref_crop_out[0]]]
                + [[c // meta["nnum"] * meta["scale"] for c in rco] for rco in ref_crop_out[1:]],
            }
        },
        {
            "Crop": {
                "apply_to": "ls_trf",
                "crop": [(0, None), (0, None), (meta["shrink"], -meta["shrink"]), (meta["shrink"], -meta["shrink"])],
            }
        },
        # {"SetPixelValue": {"apply_to": "ls_trf", "value": 1.0}},
        {
            "AffineTransformation": {
                "apply_to": {"ls_trf": "ls_trf_trf"},
                "target_to_compare_to": target_to_compare_to,
                "order": meta["interpolation_order"],
                "ref_input_shape": [838] + get_lf_shape(crop_name),
                "bdv_affine_transformations": crop_name,
                "ref_output_shape": get_ls_shape(crop_name),
                "ref_crop_in": ref_crop_in,
                "ref_crop_out": ref_crop_out,
                # "ref_crop_out": [[0, None], [0, None], [0, None]],
                # "ref_crop_out": [(17, -9), (57, -52), (86, -46)],
                "inverted": False,
                "padding_mode": "zeros",
            }
        },
    ]
    trf = get_composed_transformation_from_config(trf_config)

    ds = ZipDataset(
        OrderedDict(
            [
                ("ls_trf", get_dataset_from_info(get_tensor_info(info_name, "ls_trf", meta), cache=False)),
                ("ls", get_dataset_from_info(get_tensor_info(info_name, "ls", meta), cache=False)),
            ]
        ),
        join_dataset_masks=False,
    )

    sample = ds[0]
    sample = trf(sample)

    for name in ["ls_trf", "ls"]:
        print(name, sample[name].shape)

    import matplotlib.pyplot as plt

    def plot_vol(name):
        vol = sample[name]
        fig, ax = plt.subplots(nrows=3)
        for i in range(3):
            ax[i].imshow(vol[0, 0].max(i))
            ax[i].set_title(f"{name}_super_big{i}")

        plt.show()

    for name in ["ls_trf", "ls_trf_trf", "ls"]:
        plot_vol(name)


if __name__ == "__main__":
    test_crop()
