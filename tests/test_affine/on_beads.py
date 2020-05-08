from collections import OrderedDict

from lnet.utils import turbo_colormap

from lnet.transformations import AffineTransformation

from lnet.datasets import TensorInfo, get_dataset_from_info, ZipDataset

ref0_ls = TensorInfo(
    name="ls",
    root="GKRESHUK",
    location="LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/AffineTransforms/SwipeThrough_-450_-210_nimages_241/*Cam_Left.tif",
    insert_singleton_axes_at=[0, 0],
    tag="ref0",
)
ref0_ls_reg = TensorInfo(
    name="ls_reg",
    root="GKRESHUK",
    location="LF_partially_restored/TestOutputGcamp/LenseLeNet_Microscope/20200311_Gcamp/AffineTransforms/SwipeThrough_-450_-210_nimages_241/*Cam_Left_registered.tif",
    insert_singleton_axes_at=[0, 0],
    tag="ref0",
)


def on_beads():
    ds = ZipDataset(
        OrderedDict(
            [
                ("ls", get_dataset_from_info(ref0_ls, cache=False)),
                ("ls_reg", get_dataset_from_info(ref0_ls_reg, cache=False)),
            ]
        ),
        join_dataset_masks=False,
    )
    sample = ds[0]
    for name, tensor in sample.items():
        if name == "meta":
            continue

        print(name, tensor.shape)

    trf = AffineTransformation(
        apply_to={"ls": "ls_trf"},
        target_to_compare_to="ls_reg",
        order=2,
        ref_input_shape=(838, 1330, 1615),
        bdv_affine_transformations=[
            [
                0.98048,
                0.004709,
                0.098297,
                -111.7542,
                7.6415e-05,
                0.97546,
                0.0030523,
                -20.1143,
                0.014629,
                8.2964e-06,
                -3.9928,
                846.8515,
            ]
        ],
        ref_output_shape=(241, 1501, 1801),
        # ref_crop_in=[[152, -223], [133, 1121], [323, 1596]]  # 463x988x1273 - 0x0x0 no excess shrinkage
        # ref_crop_out=[[*ls_z_min, *ls_z_max], [152, -323], [418, -57]]
        ref_crop_in=((0, None), (0, -1), (0, -1)),  # 463x988x1273 - 0x0x0 no excess shrinkage
        ref_crop_out=((0, None), (0, None), (0, None)),
    )
    sample = trf(sample)

    import matplotlib.pyplot as plt

    def plot_vol(name):
        vol = sample[name]
        fig, ax = plt.subplots(ncols=3)
        for i in range(3):
            ax[i].imshow(vol[0, 0].max(i), cmap=turbo_colormap)
            ax[i].set_title(f"{name}{i}")
            ax[i].colorbar()

        plt.show()

    for name, tensor in sample.items():
        if name == "meta":
            continue

        plot_vol(name)


if __name__ == "__main__":
    on_beads()
