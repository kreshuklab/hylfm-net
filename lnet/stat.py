from __future__ import annotations

import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import DefaultDict, Dict, List, Optional, Sequence, Set, TYPE_CHECKING, Tuple

import numpy
import typing
import yaml

from lnet import settings

if TYPE_CHECKING:
    from numpy.lib.npyio import NpzFile
    import torch.utils.data

logger = logging.getLogger(__name__)


class DatasetStat:
    computed: dict
    requested: dict

    def __init__(
        self,
        path: Path,
        dataset: torch.utils.data.Dataset,
        percentiles: Optional[Dict[str, Set[float]]] = None,
        means: Optional[Dict[str, Set[Tuple[float, float]]]] = None,
    ):
        assert path.suffix == ".yml", path.suffix
        self.path = path
        self.dataset = dataset
        self.computed = defaultdict(dict)
        if path.exists():
            logger.warning(f"restoring computed stat from {path}")
            with path.open() as f:
                data = yaml.safe_load(f)

            def str2tuple(val: typing.Any) -> typing.Any:
                return tuple([float(s) for s in val.strip("(").strip(")").split(",")]) if isinstance(val, str) else val

            for name, comp in data.items():
                self.computed[name].update({str2tuple(key): str2tuple(val) for key, val in comp.items()})

        means = means or {}
        percentiles = percentiles or {}
        percentiles_in_means = {name: {p for range_ in ranges_ for p in range_} for name, ranges_ in means.items()}
        percentiles = {name: percentiles_in_means.get(name, set()) + p for name, p in percentiles.items()}
        self.compute_hist()
        self.compute_many_percentiles(percentiles)
        self.compute_many_mean_std(means)

    def compute_hist(self):
        n = len(self.dataset)
        sample = self.dataset[0]
        nbins = numpy.iinfo(numpy.uint16).max // 5
        hist_min = 0.0
        hist_max = numpy.iinfo(numpy.uint16).max
        self.bin_width = (hist_max - hist_min) / nbins

        hist_path = self.path.with_suffix(".hist.npz").absolute()
        if hist_path.exists():
            hist_npz: NpzFile = numpy.load(str(hist_path))
            hist = {name: hist_npz[name] for name in hist_npz.files}
        else:
            hist = {
                name: numpy.zeros(nbins, numpy.uint64)
                for name in sample.keys()
                if isinstance(sample[name], numpy.ndarray)
            }
            logger.info("Compute histograms for %s", list(hist.keys()))

            def _compute_hist(i: int):
                ret = {}
                for name, tensor in self.dataset[i].items():
                    if isinstance(tensor, numpy.ndarray):
                        ret[name] = numpy.histogram(tensor, bins=nbins, range=(hist_min, hist_max))[0]

                return ret

            if settings.max_workers_for_hist:
                futs = []
                with ThreadPoolExecutor(max_workers=settings.max_workers_for_hist) as executor:
                    for i in range(n):
                        futs.append(executor.submit(_compute_hist, i))

                    for fut in as_completed(futs):
                        for name, h in fut.result().items():
                            hist[name] = hist[name] + h  # somehow `+=` invovles casting to float64 which doesn't fly...
            else:
                for i in range(n):
                    ret = _compute_hist(i)
                    for name, h in ret.items():
                        hist[name] = hist[name] + h  # somehow `+=` invovles casting to float64 which doesn't fly...

            numpy.savez_compressed(hist_path, **hist)

        self.hist = hist
        self.cumsums = {name: h.cumsum() for name, h in hist.items()}

    def compute_many_percentiles(self, percentiles: Dict[str, Set[float]]) -> None:
        for name, pers in percentiles.items():
            pers = numpy.array(list(pers))
            per_values = numpy.searchsorted(self.cumsums[name], pers * self.cumsums[name][-1] / 100) * self.bin_width
            self.computed[name].update(dict(zip(pers.tolist(), per_values.tolist())))

        self.save_computed()

    def get_percentiles(self, name: str, percentiles: Sequence[float]) -> List[float]:
        ret = [self.computed[name].get(p, None) for p in percentiles]
        if None in ret:
            self.compute_many_percentiles({name: {p for p, ret in zip(percentiles, ret) if ret is None}})
            ret = [self.computed[name].get(p, None) for p in percentiles]

        return ret

    def get_mean_std(self, name: str, percentile_range: Tuple[float, float]) -> Tuple[float, float]:
        mean_std = self.computed[name].get(percentile_range, None)
        if mean_std:
            assert len(mean_std) == 2, mean_std
            return tuple(mean_std)

        self.get_percentiles(name, percentile_range)
        self.compute_many_mean_std({name: {percentile_range,}})
        return self.computed[name][percentile_range]

    def compute_many_mean_std(self, percentile_ranges: Dict[str, Set[Tuple[float, float]]]):
        n = len(self.dataset)
        all_new_means_vars = {
            name: {
                range_: {"means": numpy.empty((n,), dtype=numpy.float64), "vars": numpy.empty((n,), dtype=numpy.float64)}
                for range_ in ranges
            }
            for name, ranges in percentile_ranges.items()
        }

        def compute_clipped_means_vars(
            i: int
        ) -> None:
            for name, array in enumerate(self.dataset[i]):
                for range_ in percentile_ranges.get(name, []):
                    lower, upper = range_
                    if lower is not None or upper is not None:
                        array = numpy.clip(
                            array,
                            a_min=None if lower is None else self.computed[name][lower],
                            a_max=None if upper is None else self.computed[name][upper],
                        )

                    all_new_means_vars[name][range_]["means"][i] = numpy.mean(array, dtype=numpy.float64).item()
                    all_new_means_vars[name][range_]["vars"][i] = numpy.var(array, dtype=numpy.float64).item()

        if settings.max_workers_for_stat:
            with ThreadPoolExecutor(max_workers=settings.max_workers_for_stat) as executor:
                futs = [executor.submit(compute_clipped_means_vars, i=i) for i in range(n)]

                for fut in as_completed(futs):
                    exc = fut.exception()
                    if exc is not None:
                        raise exc
        else:
            [compute_clipped_means_vars(i=i) for i in range(n)]

        for name, ranges in all_new_means_vars.items():
            for range_, means_vars in ranges.items():
                means = means_vars["means"]
                mean = numpy.mean(means, dtype=numpy.float64)
                vars = means_vars["vars"]
                var = numpy.mean((vars + (means - mean) ** 2))
                self.computed[name][range_] = (float(mean), float(numpy.sqrt(var)))

        self.save_computed()

    def save_computed(self):
        def tuple2str(val: typing.Any) -> typing.Any:
            return str(val) if isinstance(val, tuple) else val

        no_tuples = {
            name: {tuple2str(key): tuple2str(val) for key, val in comp.items()} for name, comp in self.computed.items()
        }
        try:
            with self.path.open("w") as file:
                yaml.dump(no_tuples, file)
        except Exception as e:
            logger.error(e, exc_info=True)
