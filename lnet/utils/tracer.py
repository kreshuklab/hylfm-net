from pathlib import Path
from typing import Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy
import yaml
from scipy.ndimage import gaussian_filter
from skimage.feature import blob_dog, peak_local_max
from torch.utils.data import DataLoader

from lnet import settings
from lnet.datasets import TensorInfo
from lnet.datasets.base import DatasetFromInfo, TiffDataset, get_collate_fn


def trace(tgt_path: Path, tgt: str = "ls_slice", compare_to: Sequence[Union[Tuple[Path, str], str]] = ("pred",)):
    compare_to = [(tgt_path, ct) if isinstance(ct, str) else ct for ct in compare_to]
    assert all([len(ct) == 2 for ct in compare_to])
    assert all([isinstance(ct[0], Path) for ct in compare_to])
    assert all([isinstance(ct[1], str) for ct in compare_to])

    ds_tgt = TiffDataset(info=TensorInfo(name=tgt, root=tgt_path, location=f"{tgt}/*.tif"))
    length = len(ds_tgt)
    compare_to_ds = [
        TiffDataset(info=TensorInfo(name=ct[1], root=ct[0], location=f"{ct[1]}/*.tif")) for ct in compare_to
    ]
    assert all([len(ctd) == length for ctd in compare_to_ds]), [length] + [len(ctd) for ctd in compare_to_ds]

    peak_pos_figs = {}
    peak_path = tgt_path / f"{tgt}_peaks.yml"
    if False and peak_path.exists():  # todo: rm False
        with peak_path.open() as f:
            peaks = numpy.asarray(yaml.safe_load(f))
    else:
        tgt_min_path = tgt_path / f"{tgt}_min.npy"
        tgt_max_path = tgt_path / f"{tgt}_max.npy"
        if tgt_min_path.exists() and tgt_max_path.exists():
            min_tensor = numpy.load(str(tgt_min_path))
            max_tensor = numpy.load(str(tgt_max_path))
        else:
            min_tensor, max_tensor = get_min_max(ds_tgt, tgt)
            numpy.save(str(tgt_min_path), min_tensor)
            numpy.save(str(tgt_max_path), max_tensor)

        diff_tensor = max_tensor - min_tensor
        smooth_diff_tensor = gaussian_filter(diff_tensor, sigma=1.3, mode="constant")
        # diff_tensor = gaussian_filter(max_tensor, sigma=1.3, mode="constant") - gaussian_filter(min_tensor, sigma=1.3, mode="constant")
        # blobs = blob_dog(
        #     diff_tensor, min_sigma=1, max_sigma=16, sigma_ratio=1.6, threshold=.3, overlap=0.5, exclude_border=True
        # )
        # peaks = blob_dog(
        #     diff_tensor, min_sigma=1.0, max_sigma=5, sigma_ratio=1.1, threshold=.1, overlap=0.5, exclude_border=False
        # )
        # smooth_diff_tensor = diff_tensor
        # smooth_diff_tensor = gaussian_filter(diff_tensor, sigma=1.3, mode="constant")
        peaks = peak_local_max(smooth_diff_tensor, min_distance=3, threshold_abs=1.0, exclude_border=True, num_peaks=10)
        r = 6  # same radius for all
        peaks = numpy.concatenate([peaks, numpy.full((peaks.shape[0], 1), r)], axis=1)
        peaks_on = {"diff tensor": diff_tensor, "smooth diff tensor": smooth_diff_tensor}
        for name, tensor in peaks_on.items():
            # plot peak positions on smoothed diff tensor
            fig, ax = plt.subplots()
            title = f"peaks on {name}"
            ax.set_title(title)
            im = ax.imshow(diff_tensor.squeeze())
            fig.colorbar(im, ax=ax)
            for i, peak in enumerate(peaks):
                y, x, r = peak
                c = plt.Circle((x, y), r, color="r", linewidth=1, fill=False)
                plt.text(x + 2 * int(r + 0.5), y, str(i))
                ax.add_patch(c)

            ax.set_axis_off()
            plt.savefig(tgt_path / f"{tgt}_{title.replace(' ', '_')}.png")
            # plt.show()
            peak_pos_figs[name] = fig

        with peak_path.open("w") as f:
            yaml.safe_dump(peaks.tolist(), f)

    return peaks, peak_pos_figs


def get_min_max(ds: DatasetFromInfo, name: str):
    min_ = max_ = None
    for sample in DataLoader(
        dataset=ds,
        shuffle=False,
        collate_fn=get_collate_fn(lambda b: b),
        num_workers=settings.max_workers_for_trace,
        pin_memory=False,
    ):
        tensor = sample[name].squeeze()
        if min_ is None:
            min_ = tensor
            max_ = tensor
        else:
            min_ = numpy.minimum(min_, tensor)
            max_ = numpy.maximum(max_, tensor)

    assert len(min_.shape) == 2, min_.shape
    assert len(max_.shape) == 2, max_.shape
    return min_, max_


if __name__ == "__main__":
    peaks = trace(
        # tgt_path = Path("/g/kreshuk/LF_computed/lnet/logs/brain1/z_out49/f2_only11_2/20-05-19_12-27-16/test/run000/ds0-0")
        tgt_path=Path(
            "/g/kreshuk/LF_computed/lnet/logs/brain1/test_z_out49/lr_f4/20-05-31_21-24-03/brain.09_3__2020-03-09_06.43.40__SinglePlane_-330/run000/ds0-0"
        ),
        compare_to=["lr_slice"],
    )

