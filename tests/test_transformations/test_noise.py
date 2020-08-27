import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from hylfm.transformations import Normalize01, PoissonNoise


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


def test_poisson(ls_slice_dataset, add_to_tag="pytest"):
    # for p in [10, 20, 30, 40]:
    common_trfs = [
        Normalize01(apply_to=["lf"], min_percentile=5.0, max_percentile=99.8),
        Normalize01(apply_to=["ls_slice"], min_percentile=5.0, max_percentile=99.99),
    ]
    for p in [100, 200, 500]:
        sample = ls_slice_dataset[20]
        for ctrf in common_trfs:
            sample = ctrf(sample)

        sample = PoissonNoise(apply_to={"ls_slice": "ls_slice_trf", "lf": "lf_trf"}, peak=p, seed=0)(sample)
        compare_slices(sample, f"{add_to_tag}_{p}", "ls_slice", "ls_slice_trf")
        compare_slices(sample, f"{add_to_tag}_{p}", "lf", "lf_trf")

    # for p in [0, 5, 50, 99.8, 99.9, 99.99, 100]:
    #     trf = PoissonNoise(apply_to={"ls_slice": "ls_slice_trf"}, peak_percentile=p, seed=0)
    #     sample = trf(ls_slice_dataset[0])
    #     compare_slices(sample, f"percentile{p}", "ls_slice", "ls_slice_trf")


if __name__ == "__main__":
    from hylfm.datasets import ZipDataset, get_dataset_from_info, get_tensor_info

    meta = {"nnum": 19, "z_out": 49, "interpolation_order": 2, "scale": 2}
    for tag in [
        "brain.11_1__2020-03-11_03.22.33__SinglePlane_-330",
        "brain.11_2__2020-03-11_07.30.39__SinglePlane_-320",
        "brain.09_3__2020-03-09_06.43.40__SinglePlane_-330",
    ]:
        ls_slice_info = get_tensor_info(tag, "ls_slice", meta=meta)
        lf_info = get_tensor_info(tag, "lf", meta=meta)
        ls_slice_dataset = ZipDataset(
            {
                "ls_slice": get_dataset_from_info(info=ls_slice_info, cache=True),
                "lf": get_dataset_from_info(info=lf_info, cache=True),
            }
        )

        test_poisson(ls_slice_dataset, tag)
