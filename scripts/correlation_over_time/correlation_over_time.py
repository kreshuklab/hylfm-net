from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Tuple

import imageio
import matplotlib.pyplot as plt
import numpy
from scipy.ndimage import gaussian_filter
from scipy.stats import pearsonr, spearmanr

from hylfm.utils.turbo_colormap import turbo_colormap

configs = [
    {
        "root_name": "09_3__2020-03-09_06.43.40__SinglePlane_-330",
        "pred_name": "pred_only_09_3_a",
        "roi": (slice(None), slice(None)),
    },
    {
        "root_name": "09_3__2020-03-09_06.43.40__SinglePlane_-330",
        "pred_name": "pred_only_09_3_a",
        "roi": (slice(25, None), slice(95, 230)),
    },
    {
        "root_name": "09_3__2020-03-09_06.43.40__SinglePlane_-340",
        "pred_name": "pred_only_09_3_a",
        "roi": (slice(None), slice(None)),
    },
    {
        "root_name": "11_2__2020-03-11_07.30.39__SinglePlane_-310",
        "pred_name": "pred_only_11_2_a",
        "roi": (slice(50, 200), slice(70, 270)),
        "ravg": 5,
    },
    {
        "root_name": "11_2__2020-03-11_07.30.39__SinglePlane_-310",
        "pred_name": "pred_only_11_2_a",
        "roi": (slice(50, 200), slice(70, 270)),
        "ravg": 3,
    },
    {
        "root_name": "11_2__2020-03-11_07.30.39__SinglePlane_-310",
        "pred_name": "pred_only_11_2_a",
        "roi": (slice(50, 200), slice(70, 270)),
    },
    {
        "root_name": "11_2__2020-03-11_07.30.39__SinglePlane_-310",
        "pred_name": "pred_only_11_2_a",
        "roi": (slice(90, 125), slice(110, 140)),
        "tslice": slice(10, 550),
    },
    {
        "root_name": "11_2__2020-03-11_07.30.39__SinglePlane_-310",
        "pred_name": "pred_only_11_2_a",
        "roi": (slice(60, 190), slice(100, 270)),
        "tslice": slice(None),
        "smooth_ls": {"sigma": 2, "mode": "constant"},
    },
    {
        "root_name": "11_2__2020-03-11_07.30.39__SinglePlane_-310",
        "pred_name": "pred_only_11_2_a",
        "roi": (slice(60, 190), slice(100, 250)),
        "tslice": slice(None),
        "smooth_ls": {"sigma": 2, "mode": "constant"},
        "smooth_lr": {"sigma": 1, "mode": "constant"},
        "smooth_pred": {"sigma": 1, "mode": "constant"},
    },
    # {
    #     "root_name": "11_2__2020-03-11_07.30.39__SinglePlane_-310",
    #     "pred_name": "pred_only_11_2_a",
    #     "roi": (slice(90, 125), slice(110, 130)),
    # },
]


def get_img(path: Path, roi: Tuple[slice, slice]):
    return imageio.imread(path)[roi]


if __name__ == "__main__":
    run_name = f"{datetime.now().strftime('%y-%m-%d_%H-%M-%S')}"
    out_md = Path(__file__).parent / "out.md"
    with out_md.open("a") as f:
        f.write()

    print(out_md)
    assert False
    start = perf_counter()
    config = configs[-1]
    root = Path("/Users/fbeut/Desktop/lnet_stuff/manual_traces") / config["root_name"]
    ls_path = root / "ls_slice"
    lr_path = root / "lr_slice"
    pred_path = root / config["pred_name"]

    lss = []
    lrs = []
    preds = []
    all_ls_paths = sorted(ls_path.glob("*.tif"))[config.get("tslice", slice(None))]
    print("n timepoints", len(all_ls_paths))
    assert len(all_ls_paths) <= 600, len(all_ls_paths)
    for p in all_ls_paths:
        try:
            ls = get_img(p, roi=config["roi"])
            lr = get_img(lr_path / p.name, roi=config["roi"])
            pred = get_img(pred_path / p.name, roi=config["roi"])
            assert ls.shape == lr.shape == pred.shape
        except Exception as e:
            print(e)
        else:
            if config.get("smooth_ls", False):
                ls = gaussian_filter(ls, **config["smooth_ls"])

            lss.append(ls)
            lrs.append(lr)
            preds.append(pred)

            if int(p.stem) % 100 == 0:
                print(p.stem)
                fig, ax = plt.subplots(ncols=3)
                ax[0].imshow(ls, vmin=0, vmax=1, cmap=turbo_colormap)
                ax[0].set_title("SPIM")
                ax[1].imshow(lr, vmin=0, vmax=1, cmap=turbo_colormap)
                ax[1].set_title("LFM")
                ax[2].imshow(pred, vmin=0, vmax=1, cmap=turbo_colormap)
                ax[2].set_title("HyLFM")
                plt.show()

    ls = numpy.stack(lss, axis=2)
    lr = numpy.stack(lrs, axis=2)
    pred = numpy.stack(preds, axis=2)

    assert ls.shape == lr.shape == pred.shape

    print(f"loading time {perf_counter() - start:.0f}")

    pix_per_t = ls.shape[0] * ls.shape[1]
    print("shape", ls.shape, "pix_per_t", pix_per_t)

    def xyt2xt(t):
        assert len(t.shape) == 3
        return t.reshape((t.shape[0] * t.shape[1], t.shape[2]))

    ls = xyt2xt(ls)
    lr = xyt2xt(lr)
    pred = xyt2xt(pred)

    if "ravg" in config:

        def get_running_avg(time_series, r: int = config["ravg"]):
            return numpy.convolve(time_series, numpy.ones((r,)) / r, mode="valid")

        start = perf_counter()
        ls = [get_running_avg(lss) for lss in ls]
        lr = [get_running_avg(lrr) for lrr in lr]
        pred = [get_running_avg(predd) for predd in pred]
        print(f"running avg time{perf_counter() - start:.0f}")

    start = perf_counter()
    pr_lr = numpy.asarray([pearsonr(lrr, lss)[0] for lrr, lss in zip(lr, ls)])
    pr_pred = numpy.asarray([pearsonr(predd, lss)[0] for predd, lss in zip(pred, ls)])
    print(f"pearson time {perf_counter() - start:.0f}")

    print(f"pearson LFM {pr_lr.mean():.2f} std {pr_lr.std():.2f}")
    print(f"pearson HyLFM {pr_pred.mean():.2f} std {pr_pred.std():.2f}")

    start = perf_counter()
    sr_lr = numpy.asarray([spearmanr(lrr, lss)[0] for lrr, lss in zip(lr, ls)])
    sr_pred = numpy.asarray([spearmanr(predd, lss)[0] for predd, lss in zip(pred, ls)])
    print(f"spearman time {perf_counter() - start:.0f}")

    print(f"spearman   LFM {sr_lr.mean():.2f} std {sr_lr.std():.2f}")
    print(f"spearman HyLFM {sr_pred.mean():.2f} std {sr_pred.std():.2f}")

    # fig, ax = plt.subplots()
    # ax.violinplot(pr_lr)
    # ax.set_title("pr_lr")
    #
    # fig, ax = plt.subplots()
    # ax.s(pr_pred)
    # ax.set_title("pr_pred")
    #
    # fig, ax = plt.subplots()
    # ax.plot(pr_lr)
    # ax.plot(pr_pred)

    fig, ax = plt.subplots()
    ax.violinplot([pr_lr, pr_pred], showmeans=True, showmedians=True)  # quantiles=[[.25, .50, .75]]* 2
    ax.set_title("pearson")

    fig.save()
    fig, ax = plt.subplots()
    ax.violinplot([sr_lr, sr_pred], showmeans=True, showmedians=True)  # quantiles=[[.25, .50, .75]]* 2
    ax.set_title("spearman")

    # plt.show()
