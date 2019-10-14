from concurrent.futures import as_completed
from concurrent.futures.thread import ThreadPoolExecutor
from dataclasses import dataclass

import numpy
from torch.utils.data import Dataset


@dataclass
class DatasetStat:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    x_mean: float
    y_mean: float
    x_std: float
    y_std: float
    corr_x_min: float
    corr_x_max: float
    corr_y_min: float
    corr_y_max: float
    corr_x_mean: float
    corr_y_mean: float
    corr_x_std: float
    corr_y_std: float


def compute_stat(ds: Dataset) -> DatasetStat:
    # samples in ds are required to have the same weight

    n = len(ds)

    x_mins = numpy.empty((n,), dtype=numpy.float32)
    x_maxs = numpy.empty((n,), dtype=numpy.float32)
    x_means = numpy.empty((n,), dtype=numpy.float64)
    x_vars = numpy.empty((n,), dtype=numpy.float64)

    y_mins = numpy.empty((n,), dtype=numpy.float32)
    y_maxs = numpy.empty((n,), dtype=numpy.float32)
    y_means = numpy.empty((n,), dtype=numpy.float64)
    y_vars = numpy.empty((n,), dtype=numpy.float64)

    corr_x_mins = numpy.empty((n,), dtype=numpy.float32)
    corr_x_maxs = numpy.empty((n,), dtype=numpy.float32)
    corr_x_means = numpy.empty((n,), dtype=numpy.float64)
    corr_x_vars = numpy.empty((n,), dtype=numpy.float64)

    corr_y_mins = numpy.empty((n,), dtype=numpy.float32)
    corr_y_maxs = numpy.empty((n,), dtype=numpy.float32)
    corr_y_means = numpy.empty((n,), dtype=numpy.float64)
    corr_y_vars = numpy.empty((n,), dtype=numpy.float64)

    def get_stat_idx(i: int):
        x, y = ds[i]
        x_mins[i] = numpy.nanpercentile(x, 0.05)
        x_maxs[i] = numpy.nanpercentile(x, 99.95)
        x_means[i] = numpy.nanmean(x)
        x_vars[i] = numpy.nanvar(x)

        y_mins[i] = numpy.nanpercentile(y, 0.05)
        y_maxs[i] = numpy.nanpercentile(y, 99.95)
        y_means[i] = numpy.nanmean(y)
        y_vars[i] = numpy.nanvar(y)

        corr_x_mins[i] = numpy.nanpercentile(x, 6.65)
        x = x[x > corr_x_mins[i]]
        corr_x_maxs[i] = numpy.nanpercentile(x, 99.95)
        x = x[x < corr_x_maxs[i]]
        # x = numpy.clip(x, corr_x_mins[i], corr_x_maxs[i])
        corr_x_means[i] = numpy.nanmean(x)
        corr_x_vars[i] = numpy.nanvar(x)

        y_std = numpy.nanstd(y)
        corr_y_mins[i] = numpy.nanmedian(y)
        y = y[y > corr_y_mins[i] + 5 * y_std]
        corr_y_maxs[i] = numpy.nanpercentile(y, 99.95)
        y = y[y < corr_y_maxs[i]]
        # y = numpy.clip(y, corr_y_mins[i], corr_y_maxs[i])
        corr_y_means[i] = numpy.nanmean(y)
        corr_y_vars[i] = numpy.nanvar(y)

    # for i in range(n):
    #     get_stat_idx(i)
    #
    # print('x_mins', x_mins)
    # print('x_maxs', x_maxs)
    # print('x_means', x_means)
    # print('x_vars', x_vars)
    # print('y_mins', y_mins)
    # print('y_maxs', y_maxs)
    # print('y_means', y_means)
    # print('y_vars', y_vars)

    futs = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        for i in range(n):
            futs.append(executor.submit(get_stat_idx, i))

        for fut in as_completed(futs):
            e = fut.exception()
            if e is not None:
                raise e

    x_mean = numpy.mean(x_means)
    x_var = numpy.mean((x_vars + (x_means - x_mean) ** 2))

    y_mean = numpy.mean(y_means)
    y_var = numpy.mean((y_vars + (y_means - y_mean) ** 2))

    corr_x_mean = numpy.mean(corr_x_means)
    corr_x_var = numpy.mean((corr_x_vars + (corr_x_means - corr_x_mean) ** 2))

    corr_y_mean = numpy.mean(corr_y_means)
    corr_y_var = numpy.mean((corr_y_vars + (corr_y_means - corr_y_mean) ** 2))

    return DatasetStat(
        x_min=float(numpy.mean(x_mins)),
        x_max=float(numpy.mean(x_maxs)),
        x_mean=float(x_mean),
        x_std=float(numpy.sqrt(x_var)),
        y_min=float(numpy.mean(y_mins)),
        y_max=float(numpy.mean(y_maxs)),
        y_mean=float(y_mean),
        y_std=float(numpy.sqrt(y_var)),
        corr_x_min=float(numpy.mean(corr_x_mins)),
        corr_x_max=float(numpy.mean(corr_x_maxs)),
        corr_x_mean=float(corr_x_mean),
        corr_x_std=float(numpy.sqrt(corr_x_var)),
        corr_y_min=float(numpy.mean(corr_y_mins)),
        corr_y_max=float(numpy.mean(corr_y_maxs)),
        corr_y_mean=float(corr_y_mean),
        corr_y_std=float(numpy.sqrt(corr_y_var)),
    )
