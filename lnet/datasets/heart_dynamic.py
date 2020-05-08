from pathlib import Path

from lnet.datasets.base import TensorInfo, get_dataset_from_info
from lnet.datasets.heart_utils import get_transformations, idx2z_slice_241


def get_tensor_info(tag: str, name: str, meta: dict):
    root = "GKRESHUK"
    insert_singleton_axes_at = [0, 0]
    z_slice = None
    samples_per_dataset = 1

    if "_repeat" in name:
        name, repeat = name.split("_repeat")
        repeat = int(repeat)
    else:
        repeat = 1

    if tag in ["2019-12-02_23.17.56", "2019-12-02_23.43.24", "2019-12-02_23.50.04", "2019-12-03_00.00.44"]:
        location = f"LF_partially_restored/LenseLeNet_Microscope/20191203_dynamic_staticHeart_tuesday/fish1/dynamic/Heart_tightCrop/dynamicImaging1_btw20to160planes/{tag}/stack_1_channel_3/"
        if tag == "2019-12-03_00.00.44":
            location = location.replace("stack_1_channel_3", "stack_1_channel_2")

        if tag in ["2019-12-02_23.43.24", "2019-12-02_23.50.04", "2019-12-03_00.00.44"]:
            if name == "lf":
                if tag != "2019-12-03_00.00.44":
                    location += "originalCrop/"

                padding = [
                    {
                        "Pad": {
                            "apply_to": name,
                            "pad_width": [[0, 0], [0, 1], [0, 0]],
                            "pad_mode": "lenslets",
                            "nnum": meta["nnum"],
                        }
                    }
                ]
            elif name == "lr":
                if tag != "2019-12-03_00.00.44":
                    location += "originalCrop/"

                raise NotImplementedError("padding for lr")
            else:
                padding = []
        else:
            padding = []

        transformations = padding + get_transformations(name, "Heart_tightCrop", meta=meta)
        if name == "lf":
            location += "TP_*/RC_rectified/Cam_Right_*_rectified.tif"
        # elif name == "lr":
        #     location = location.replace("LF_partially_restored/", "LF_computed/")
        #     location += "TP_*/RCout/Cam_Right_001.tif"
        elif name == "ls_slice":
            location += "Cam_Left_*.h5/Data"
            samples_per_dataset = 241
            z_slice = idx2z_slice_241

    elif tag in [
        "2019-12-09_23.10.02",
        "2019-12-09_23.17.30",
        "2019-12-09_23.19.41",
        "2019-12-10_00.40.09",
        "2019-12-10_00.51.54",
        "2019-12-10_01.03.50",
        "2019-12-10_01.25.44",
    ]:
        transformations = get_transformations(name, "Heart_tightCrop", meta=meta)
        location = f"LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish3/dynamic/Heart_tightCrop/slideThroughStack/{tag}/"
        if name == "lf":
            location += "stack_1_channel_3/TP_*/RC_rectified/Cam_Right_*_rectified.tif"
        # elif name == "lr":
        #     location = location.replace("LF_partially_restored/", "LF_computed/")
        #     location += "stack_1_channel_3/TP_*/RCout/Cam_Right_001.tif"
        elif name == "ls_slice":
            location += "stack_1_channel_3/Cam_Left_*.h5/Data"
            samples_per_dataset = 241
            z_slice = idx2z_slice_241

    elif tag in ["2019-12-09_04.54.38", "2019-12-09_05.21.16"]:
        transformations = get_transformations(name, "Heart_tightCrop", meta=meta)
        location = f"LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/{tag}/"
        if name == "lf":
            location += "stack_1_channel_3/TP_*/RC_rectified/Cam_Right_*_rectified.tif"
        # elif name == "lr":
        #     location = location.replace("LF_partially_restored/", "LF_computed/")
        #     location += "stack_1_channel_3/TP_*/RCout/Cam_Right_001.tif"
        elif name == "ls_slice":
            location += "stack_1_channel_3/Cam_Left_*.h5/Data"
            samples_per_dataset = 241
            z_slice = idx2z_slice_241

    elif tag in ["2019-12-08_23.43.42"]:
        transformations = get_transformations(name, "Heart_tightCrop", meta=meta)
        location = f"LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish1/dynamic/Heart_tightCrop/SlideThroughCompleteStack/{tag}/"
        if name == "lf":
            location += "stack_1_channel_3/TP_*/RC_rectified/Cam_Right_*_rectified.tif"
        # elif name == "lr":
        #     location = location.replace("LF_partially_restored/", "LF_computed/")
        #     location += "stack_1_channel_3/TP_*/RCout/Cam_Right_001.tif"
        elif name == "ls_slice":
            location += "stack_1_channel_3/Cam_Left_*.h5/Data"
            samples_per_dataset = 241
            z_slice = idx2z_slice_241

    # test data candidates...
    elif tag in ["2019-12-02_04.12.36_10msExp"]:  # todo: double check if this fits to 'Heart_tigthCrop'
        transformations = get_transformations(name, "Heart_tightCrop", meta=meta)
        location = f"LF_partially_restored/LenseLeNet_Microscope/20191202_staticHeart_dynamicHeart/data/{tag}/"
        if name == "lf":
            location += "stack_1_channel_3/TP_*/RC_rectified/Cam_Right_*_rectified.tif"
        # elif name == "lr":
        #     location = location.replace("LF_partially_restored/", "LF_computed/")
        #     location += "stack_1_channel_3/TP_*/RCout/Cam_Right_001.tif"
        elif name == "ls_slice":
            location += "stack_1_channel_3/Cam_Left_*.h5/Data"
            samples_per_dataset = 241
            z_slice = idx2z_slice_241

    elif tag in ["2019-12-09_07.50.24"]:  # todo: check if really dynamic
        transformations = get_transformations(name, "Heart_tightCrop", meta=meta)
        location = f"LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.41.14_theGoldenOne/staticHeart_samePos/{tag}/"
        if name == "lf":
            location += "stack_1_channel_3/TP_*/RC_rectified/Cam_Right_*_rectified.tif"
        # elif name == "lr":
        #     location = location.replace("LF_partially_restored/", "LF_computed/")
        #     location += "stack_1_channel_3/TP_*/RCout/Cam_Right_001.tif"
        elif name == "ls_slice":
            location += "stack_1_channel_3/Cam_Left_*.h5/Data"
            samples_per_dataset = 241
            z_slice = idx2z_slice_241

    elif tag in ["2019-12-09_05.41.14_theGoldenOne"]:
        transformations = get_transformations(name, "Heart_tightCrop", meta=meta)
        location = f"LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/{tag}/"
        if name == "lf":
            location += "stack_1_channel_3/TP_*/RC_rectified/Cam_Right_*_rectified.tif"
        # elif name == "lr":
        #     location = location.replace("LF_partially_restored/", "LF_computed/")
        #     location += "stack_1_channel_3/TP_*/RCout/Cam_Right_001.tif"
        elif name == "ls_slice":
            location += "stack_1_channel_3/Cam_Left_*.h5/Data"
            samples_per_dataset = 241
            z_slice = idx2z_slice_241

    elif tag in ["2019-12-09_05.55.26"]:
        transformations = get_transformations(name, "Heart_tightCrop", meta=meta)
        location = f"LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/dynamic/Heart_tightCrop/2019-12-09_05.41.14_theGoldenOne/singlePlane_samePos/plane_100/{tag}/"
        if name == "lf":
            location += "stack_2_channel_3/TP_*/RC_rectified/Cam_Right_*_rectified.tif"
        # elif name == "lr":
        #     location = location.replace("LF_partially_restored/", "LF_computed/")
        #     location += "stack_1_channel_3/TP_*/RCout/Cam_Right_001.tif"
        elif name == "ls_slice":
            location += "stack_2_channel_3/Cam_Left_*.h5/Data"
            samples_per_dataset = 200
            z_slice = 140
            assert z_slice == idx2z_slice_241(100), (z_slice, idx2z_slice_241(100))
    else:
        raise NotImplementedError(tag)

    if location is None or location.endswith("/"):
        raise NotImplementedError(f"tag: {tag}, name: {name}")

    assert tag in location, (tag, name, location)
    return TensorInfo(
        name=name,
        root=root,
        location=location,
        insert_singleton_axes_at=insert_singleton_axes_at,
        transformations=transformations,
        z_slice=z_slice,
        samples_per_dataset=samples_per_dataset,
        repeat=repeat,
        tag=tag.replace("-", "").replace(".", ""),
        meta=meta,
    )


def debug():
    tight_heart_bdv = [
        [
            0.97945,
            0.0048391,
            -0.096309,
            -88.5296,
            -0.0074754,
            0.98139,
            0.15814,
            -91.235,
            0.016076,
            0.0061465,
            4.0499,
            -102.0931,
        ]
    ]

    def get_vol_trf(name: str):
        return [
            {
                "Resize": {
                    "apply_to": name,
                    "shape": [1.0, 1.0, 0.21052631578947368421052631578947, 0.21052631578947368421052631578947],
                    # "shape": [1.0, 1.0, 0.42105263157894736842105263157895, 0.42105263157894736842105263157895],
                    "order": 2,
                }
            }
        ]

    # ds_ls = get_dataset_from_info(f20191209_081940_ls, transformations=get_vol_trf("ls"), cache=True, indices=[0])
    # print("len ls", len(ds_ls))

    # ds_ls_trf = get_dataset_from_info(
    #     f20191209_081940_ls_trf, transformations=get_vol_trf("ls_trf"), cache=True, indices=[0]
    # )
    # print("len ls_trf", len(ds_ls_trf))
    # ds_ls_trf = get_dataset_from_info(
    #     f20191209_081940_ls_trf, transformations=[], cache=True, indices=[0]
    # )

    # ds_ls_tif = get_dataset_from_info(
    #     f20191209_081940_ls_tif, transformations=get_vol_trf("ls_tif"), cache=True, indices=[0]
    # )
    # print("len ls tif", len(ds_ls_tif))

    # slice_indices = [0, 1, 2, 3, 4, 40, 80, 120, 160, 200, 240]
    # ds_ls_slice = get_dataset_from_info(
    #     f20191209_081940_ls_slice, transformations=get_vol_trf("ls_slice"), cache=True, indices=slice_indices
    # )

    # print("len ls_tif", len(ds))
    # ls_tif = ds[0]["ls_tif"]
    # print("ls_tif", ls_tif.shape)
    #
    # print("diff max", ls.max(), ls_tif.max())
    # print("max diff", (ls - ls_tif).max())
    #
    #
    # plt.imshow(ls[0, 0].max(0))
    # plt.title("ls0")
    # plt.show()
    # plt.imshow(ls[0, 0].max(1))
    # plt.title("ls1")
    # plt.show()
    #
    # plt.imshow(ls_tif[0, 0].max(0))
    # plt.title("ls_tif0")
    # plt.show()
    # plt.imshow(ls_tif[0, 0].max(1))
    # plt.title("ls_tif1")
    # plt.show()

    # ds_lr = get_dataset_from_info(f20191209_081940_lr, transformations=get_vol_trf("lr"), cache=False, indices=[0])
    # print("len ds_lr", len(ds_lr))
    # ds_lr_repeat = get_dataset_from_info(
    #     f20191209_081940_lr_repeat, transformations=get_vol_trf("lr"), cache=True, indices=slice_indices
    # )
    # print("len ds_lr_repeat", len(ds_lr_repeat))

    # ds_zip_slice = ZipDataset(
    #     datasets={"lr": ds_lr_repeat, "ls_slice": ds_ls_slice},
    #     transformation=AffineTransformation(
    #         apply_to={"lr": "lr_trf"},
    #         target_to_compare_to="ls_slice",
    #         order=2,
    #         ref_input_shape=[838, 1273, 1463],
    #         bdv_affine_transformations=tight_heart_bdv,
    #         ref_output_shape=[241, 1451, 1651],
    #         ref_crop_in=[[0, None], [0, None], [0, None]],
    #         ref_crop_out=[[0, None], [0, None], [0, None]],
    #         inverted=False,
    #         padding_mode="border",
    #     ),
    # )

    # ds_zip = ZipDataset(
    #     datasets={"ls": ds_ls, "lr": ds_lr, "ls_tif": ds_ls_tif},
    #     # transformation=AffineTransformation(
    #     #     apply_to={"lr": "lr_trf"},
    #     #     target_to_compare_to="ls",
    #     #     order=2,
    #     #     ref_input_shape=[838, 1273, 1463],
    #     #     bdv_affine_transformations=tight_heart_bdv,
    #     #     ref_output_shape=[241, 1451, 1651],
    #     #     ref_crop_in=[[0, None], [0, None], [0, None]],
    #     #     ref_crop_out=[[0, None], [0, None], [0, None]],
    #     #     inverted=False,
    #     #     padding_mode="border",
    #     # ),
    # )
    # sample = ds_zip[0]

    # ds_zip = ZipDataset(datasets={"ls_trf": ds_ls_trf, "lr": ds_lr})
    # sample = ds_zip[0]

    # def save_vol(name):
    #     vol = sample[name]
    #     imageio.volwrite(f"/g/kreshuk/LF_computed/lnet/debug_affine_fish_trf/{name}.tif", numpy.squeeze(vol))
    #
    # def plot_vol(name):
    #     vol = sample[name]
    #     fig, ax = plt.subplots(2)
    #     for i in range(2):
    #         ax[i].imshow(vol[0, 0].max(i))
    #         ax[i].set_title(f"{name}_super_big{i}")
    #
    #     plt.show()
    #
    # for name in ["ls_trf"]:
    #     save_vol(name)
    #     plot_vol(name)

    # for idx in range(11):
    #     sample = ds_zip_slice[idx]
    #     fig, ax = plt.subplots(2)
    #     ax[0].imshow(sample["ls_slice"][0, 0, 0])
    #     ax[0].set_title(f"ls_slice idx: {idx} z: {sample['meta'][0]['ls_slice']['z_slice']}")
    #     ax[1].imshow(sample["lr_trf"][0, 0, 0])
    #     ax[1].set_title(f"lr_trf idx: {idx}")
    #     plt.show()


def check_data():
    meta = {"z_out": 49, "nnum": 19, "scale": 4}

    # 2019-12-02_23.43.24 not matching! -> use 2019-12-02_23.43.24/stack_1_channel_3/originalCrop/TP_00000_originalCrop/ with 19px padding on rectified image on bottom (bead IMG is bigger)
    # 2019-12-02_23.50.04 not matching! -> use 2019-12-02_23.50.04/stack_1_channel_3/originalCrop/TP_00000/ with 19px padding on rectified image on bottom (bead IMG is bigger)
    # 2019-12-03_00.00.44/stack_1_channel_3/ not matching! -> use 2019-12-03_00.00.44/stack_1_channel_2/ with 19px padding on rectified image on bottom (bead IMG is bigger)

    for tag in [
        "2019-12-02_23.17.56",
        "2019-12-02_23.43.24",  # were too small, now ok
        "2019-12-02_23.50.04",   # were too small, now ok
        "2019-12-03_00.00.44",  # were too small, now ok
        "2019-12-09_23.10.02",
        "2019-12-09_23.17.30",
        "2019-12-09_23.19.41",
        "2019-12-10_00.40.09",
        "2019-12-10_00.51.54",
        "2019-12-10_01.03.50",
        "2019-12-10_01.25.44",
        "2019-12-09_04.54.38",
        "2019-12-09_05.21.16",
        "2019-12-08_23.43.42",
        # "2019-12-02_04.12.36_10msExp",  # ValueError: expected shape [1, 1, 1273, 1463], but found (1, 1, 1083, 1083)
        "2019-12-09_07.50.24",
        "2019-12-09_05.41.14_theGoldenOne",
        "2019-12-09_05.55.26",
    ]:
        lf = get_dataset_from_info(get_tensor_info(tag, "lf", meta=meta))
        ls = get_dataset_from_info(get_tensor_info(tag, "ls_slice", meta=meta))
        assert len(lf) == len(ls), (tag, len(lf), len(ls))
        assert len(lf) > 0, tag
        print(tag, len(lf))
        lf = lf[0]["lf"]
        ls = ls[0]["ls_slice"]
        print("\tlf", lf.shape)
        print("\tls", ls.shape)
        # imageio.imwrite(f"/g/kreshuk/LF_computed/lnet/padded_lf_{tag}_pad_at_1.tif", lf[0, 0])
        # path = Path(f"/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191203_dynamic_staticHeart_tuesday/fish1/dynamic/Heart_tightCrop/dynamicImaging1_btw20to160planes/{tag}/stack_1_channel_3/originalCrop/TP_00000/RC_rectified_padded/Cam_Right_001_rectified.tif")
        # path.parent.mkdir(parents=True, exist_ok=True)
        # imageio.imwrite(path, lf[0, 0])

def search_data():
    path = Path(
        "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191203_dynamic_staticHeart_tuesday/fish1/dynamic/Heart_tightCrop/dynamicImaging1_btw20to160planes/2019-12-03_00.00.44/stack_1_channel_2"
    )
    for dir in path.glob("*/RC_rectified/"):
        print(dir.parent.name, len(list(dir.glob("*.tif"))))


if __name__ == "__main__":
    # depug()
    check_data()
    # search_data()