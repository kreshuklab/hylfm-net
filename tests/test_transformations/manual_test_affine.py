import collections

from hylfm.datasets import ZipDataset, get_dataset_from_info, get_tensor_info

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def compare_slices(sample, title, *names):
    fig, axes = plt.subplots(ncols=len(names))
    fig.suptitle(title)
    for name, ax in zip(names, axes):
        im = ax.imshow(sample[name].squeeze())
        ax.set_title(f"{name}")
        # fig.colorbar(im, cax=ax, orientation='horizontal')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax, orientation="vertical")

    plt.show()


def manual_test_this():
    meta = {
        "nnum": 19,
        "z_out": 49,
        "scale": 4,
        "shrink": 8,
        "interpolation_order": 2,
        "z_ls_rescaled": 241,
        "pred_z_min": 0,
        "pred_z_max": 838,
            "crop_names": ["wholeFOV"],
    }  # z_min full: 0, z_max full: 838; 60/209*838=241; 838-10/209*838=798
    ls_info = get_tensor_info("heart_static.beads_ref_wholeFOV", "ls", meta=meta)
    ls_trf_info = get_tensor_info("heart_static.beads_ref_wholeFOV", "ls_trf", meta=meta)
    ls_reg_info = get_tensor_info("heart_static.beads_ref_wholeFOV", "ls_reg", meta=meta)
    dataset = ZipDataset(
        collections.OrderedDict(
            [
                ("ls", get_dataset_from_info(info=ls_info, cache=True)),
                ("ls_trf", get_dataset_from_info(info=ls_trf_info, cache=True)),
                ("ls_reg", get_dataset_from_info(info=ls_reg_info, cache=True)),
            ]
        )
    )
    sample = dataset[0]
    compare_slices({"ls_reg": sample["ls_reg"].max(2), "ls_trf": sample["ls_trf"].max(2)}, "lala", "ls_reg", "ls_trf")


if __name__ == "__main__":
    manual_test_this()
