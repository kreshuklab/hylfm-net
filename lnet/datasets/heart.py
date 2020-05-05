from lnet.datasets import ZipDataset
from lnet.datasets.base import TensorInfo, get_dataset_from_info


def get_transformations(name: str):
    if name == "lf":
        return [{"Assert": {"apply_to": "lf", "expected_tensor_shape": [1, 1, 1451, 1651]}}]
    elif name == "ls":
        return [
            {"Assert": {"apply_to": "ls", "expected_tensor_shape": [1, 1, 241, 2048, 2060]}},
            {"FlipAxis": {"apply_to": "ls", "axis": 2}},
            {"FlipAxis": {"apply_to": "ls", "axis": 1}},
            {
                "Crop": {"apply_to": "ls", "crop": [[0, None], [0, None], [249, 1700], [199, 1850]]}
            },  # in matlab: 200, 250, 1650, 1450
            {"Assert": {"apply_to": "ls", "expected_tensor_shape": [1, 1, 241, 1451, 1651]}},
        ]
    elif name == "ls_slice":
        return [
            {"Assert": {"apply_to": "ls_slice", "expected_tensor_shape": [1, 1, 1, 2048, 2060]}},
            {"FlipAxis": {"apply_to": "ls_slice", "axis": 2}},
            {
                "Crop": {"apply_to": "ls_slice", "crop": [[0, None], [0, None], [249, 1700], [199, 1850]]}
            },  # in matlab: 200, 250, 1650, 1450
            {"Assert": {"apply_to": "ls_slice", "expected_tensor_shape": [1, 1, 1, 1451, 1651]}},
        ]
    elif name == "ls_trf":
        return [
            {"Assert": {"apply_to": "ls_trf", "expected_tensor_shape": [1, 1, 241, 2048, 2060]}},
            {"FlipAxis": {"apply_to": "ls_trf", "axis": 2}},
            {"FlipAxis": {"apply_to": "ls_trf", "axis": 1}},
            {
                "Crop": {"apply_to": "ls_trf", "crop": [[0, None], [0, None], [249, 1700], [199, 1850]]}
            },  # in matlab: 200, 250, 1650, 1450
            {"Assert": {"apply_to": "ls_trf", "expected_tensor_shape": [1, 1, 241, 1451, 1651]}},
            {
                "AffineTransformation": {
                    "apply_to": "ls_trf",
                    "target_to_compare_to": [838, 1273, 1463],  # todo: adapt
                    "order": 2,
                    "ref_input_shape": [838, 1273, 1463],
                    "bdv_affine_transformations": [
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
                    ],
                    "ref_output_shape": [241, 1451, 1651],
                    "ref_crop_in": [[0, None], [0, None], [0, None]],
                    "ref_crop_out": [[0, None], [0, None], [0, None]],
                    "inverted": True,
                    "padding_mode": "border",
                }
            },
        ]
    elif name == "lr":
        return []
    else:
        raise NotImplementedError(name)


def idx2z_slice_241(idx: int) -> int:
    return 240 - (idx % 241)


def get_tensor_info(tag: str, name: str):
    root = "GKRESHUK"
    insert_singleton_axes_at = [0, 0]
    transformations = get_transformations(name)
    location = None
    z_slice = None
    samples_per_dataset = 1

    if "_repeat" in name:
        name, repeat = name.split("_repeat")
        repeat = int(repeat)
    else:
        repeat = 1

    if tag in ["2019-12-09_08.15.07", "2019-12-09_08.19.40", "2019-12-09_08.27.14"]:
        location = f"LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/static/Heart_tightCrop/5steps_stepsize8/{tag}/"
        if name == "lf":
            location += "stack_3_channel_0/TP_*/RC_rectified/Cam_Right_001_rectified.tif"
        elif name == "lr":
            location = location.replace("LF_partially_restored/", "LF_computed/")
            location += "stack_3_channel_0/TP_*/RCout/Cam_Right_001.tif"
        elif name == "ls" or name == "ls_trf":
            location += "stack_4_channel_1/Cam_Left_*.h5/Data"
        elif name == "ls_fake_slice":
            location += "stack_4_channel_1/Cam_Left_*.h5/Data"
            samples_per_dataset = 241
            z_slice = idx2z_slice_241
    elif tag in [
        "2019-12-09_08.34.44",
        "2019-12-09_08.41.41",
        "2019-12-09_08.51.01",
        "2019-12-09_09.01.28",
        "2019-12-09_09.11.59",
        "2019-12-09_09.18.01",
    ]:
        location = f"LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/static/Heart_tightCrop/5steps_stepsize15/{tag}/"
        if name == "lf":
            location += "stack_3_channel_0/TP_*/RC_rectified/Cam_Right_001_rectified.tif"
        elif name == "lr":
            location = location.replace("LF_partially_restored/", "LF_computed/")
            location += "stack_3_channel_0/stack_3_channel_0/TP_*/RCout/Cam_Right_001.tif"
        elif name == "ls" or name == "ls_trf":
            location += "stack_4_channel_1/Cam_Left_*.h5/Data"
        elif name == "ls_fake_slice":
            location += "stack_4_channel_1/Cam_Left_*.h5/Data"
            samples_per_dataset = (241,)
            z_slice = idx2z_slice_241

    if location is None or location.endswith("/"):
        raise NotImplementedError(f"tag: {tag}, name: {name}")

    assert tag[1:] in location.replace("-", "").replace(".", ""), (tag, location)
    return TensorInfo(
        name=name,
        root=root,
        location=location,
        insert_singleton_axes_at=insert_singleton_axes_at,
        transformations=transformations,
        z_slice=z_slice,
        samples_per_dataset=samples_per_dataset,
        repeat=repeat,
    )


def debug():
    import imageio
    import matplotlib.pyplot as plt
    import numpy

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
    for tag in [
        "f20191209_081507",
        "f20191209_081940",
        "f20191209_082714",
        "f20191209_083444",
        "f20191209_084141",
        "f20191209_085101",
        "f20191209_090128",
        "f20191209_091159",
        "f20191209_091801",
    ]:
        print(tag)
        lf = get_dataset_from_info(get_tensor_info(tag, "lf"))
        ls = get_dataset_from_info(get_tensor_info(tag, "ls"))
        assert len(lf) == len(ls), (tag, len(lf), len(ls))


if __name__ == "__main__":
    # depug()
    check_data()
