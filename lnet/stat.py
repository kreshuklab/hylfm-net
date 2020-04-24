from __future__ import annotations

import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import DefaultDict, Dict, List, Optional, Sequence, Set, TYPE_CHECKING, Tuple

import numpy
import yaml
# from torch.multiprocessing import RLock

from lnet import settings

if TYPE_CHECKING:
    from numpy.lib.npyio import NpzFile
    import torch.utils.data

logger = logging.getLogger(__name__)


@dataclass
class ComputedDatasetStat:
    all_percentiles: DefaultDict[str, Dict[float, float]] = field(default_factory=lambda: defaultdict(dict))
    all_mean_std_by_percentile_range: DefaultDict[str, Dict[Tuple[float, float], Tuple[float, float]]] = field(
        default_factory=lambda: defaultdict(dict)
    )


@dataclass
class RequestedDatasetStat:
    all_percentiles: DefaultDict[str, Set[float]] = field(default_factory=lambda: defaultdict(set))
    all_mean_std_by_percentile_range: DefaultDict[str, Set[Tuple[float, float]]] = field(
        default_factory=lambda: defaultdict(set)
    )


# DatasetStat_rlock = RLock()


class DatasetStat:
    computed: ComputedDatasetStat
    requested: RequestedDatasetStat

    def __init__(
        self,
        path: Path,
        dataset: torch.utils.data.Dataset,
        percentiles: Optional[Dict[str, Set[float]]] = None,
        means: Optional[Dict[str, Set[Tuple[float, float]]]] = None,
    ):
        assert path.suffix == ".yml", path.suffix == "yaml"
        self.path = path
        self.dataset = dataset
        if path.exists():
            with path.open() as f:
                loaded_data = yaml.load(f, Loader=yaml.UnsafeLoader)
                for as_default_dict in ["all_percentiles", "all_mean_std_by_percentile_range"]:
                    loaded_data[as_default_dict] = defaultdict(dict, loaded_data.pop(as_default_dict))

                self.computed = ComputedDatasetStat(**loaded_data)
        else:
            self.computed = ComputedDatasetStat()

        if percentiles is None:
            percentiles = {}

        if means is None:
            means = {}

        self.requested = RequestedDatasetStat(
            all_percentiles=defaultdict(set, **percentiles), all_mean_std_by_percentile_range=defaultdict(set, **means)
        )
        self.compute_hist()

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

            def compute_hist(i: int):
                ret = {}
                for name, tensor in self.dataset[i].items():
                    if isinstance(tensor, numpy.ndarray):
                        ret[name] = numpy.histogram(tensor, bins=nbins, range=(hist_min, hist_max))[0]

                return ret

            futs = []
            with ThreadPoolExecutor(max_workers=settings.max_workers_for_hist) as executor:
                for i in range(n):
                    futs.append(executor.submit(compute_hist, i))

                for fut in as_completed(futs):
                    for name, h in fut.result().items():
                        hist[name] = hist[name] + h  # somehow `+=` invovles casting to float64 which doesn't fly...
            # for i in range(n):
            #     ret = compute_hist(i)
            #     for name, h in ret.items():
            #         hist[name] = hist[name] + h  # somehow `+=` invovles casting to float64 which doesn't fly...

            numpy.savez_compressed(hist_path, **hist)

        self.hist = hist
        self.cumsums = {name: h.cumsum() for name, h in hist.items()}

    def request(
        self,
        all_percentiles: Optional[Dict[str, Set[float]]] = None,
        all_mean_std_by_percentile_range: Optional[Dict[str, Set[Tuple[float, float]]]] = None,
    ):
        # with DatasetStat_rlock:
        if all_mean_std_by_percentile_range:
            for idx, pranges in all_mean_std_by_percentile_range.items():
                self.requested.all_mean_std_by_percentile_range[idx] |= pranges - set(
                    self.computed.all_mean_std_by_percentile_range[idx]
                )

                # also request all involved percentiles
                req = set()
                for pr in pranges:
                    req |= set(pr)

                self.requested.all_percentiles[idx] |= req - set(self.computed.all_percentiles[idx])

        if all_percentiles:
            for idx, pers in all_percentiles.items():
                self.requested.all_percentiles[idx] |= pers - set(self.computed.all_percentiles[idx])

    def compute_requested(self):
        # with DatasetStat_rlock:
        n = len(self.dataset)
        all_new_percentiles: Dict[str, Dict[float, float]] = {}
        for name, pers in self.requested.all_percentiles.items():
            pers = numpy.array(list(pers))
            per_values = numpy.searchsorted(self.cumsums[name], pers * self.cumsums[name][-1] / 100) * self.bin_width
            all_new_percentiles[name] = dict(zip(pers.tolist(), per_values.tolist()))

        for name, new_percentiles in all_new_percentiles.items():
            self.computed.all_percentiles[name].update(new_percentiles)

        def compute_clipped_means_vars(
            i: int, ranges: DefaultDict[str, Set[Tuple[float, float]]]
        ) -> Tuple[int, DefaultDict[str, Dict[Tuple[float, float], Tuple[float, float]]]]:
            ret = defaultdict(dict)
            for name, data in enumerate(self.dataset[i]):
                for range_ in ranges[name]:
                    lower, upper = range_
                    if lower is not None or upper is not None:
                        data = numpy.clip(
                            data,
                            a_min=None if lower is None else self.computed.all_percentiles[name][lower],
                            a_max=None if upper is None else self.computed.all_percentiles[name][upper],
                        )

                    ret[name][range_] = (
                        numpy.mean(data, dtype=numpy.float64).item(),
                        numpy.var(data, dtype=numpy.float64).item(),
                    )

            return i, ret

        all_new_means_vars: DefaultDict[
            str, DefaultDict[Tuple[float, float]], Tuple[numpy.array, numpy.array]
        ] = defaultdict(
            lambda: defaultdict(
                lambda: (numpy.empty((n,), dtype=numpy.float64), numpy.empty((n,), dtype=numpy.float64))
            )
        )
        # with ThreadPoolExecutor(max_workers=16) as executor:
        #     futs = [
        #         executor.submit(
        #             compute_clipped_means_vars, i=i, ranges=self.requested.all_mean_std_by_percentile_range
        #         )
        #         for i in range(n)
        #     ]
        #
        #     for fut in as_completed(futs):
        #         fut_res: Tuple[int, DefaultDict[str, Dict[Tuple[float, float], Tuple[float, float]]]] = fut.result()
        #         i, res = fut_res
        #         for name, means_dict in res.items():
        #             for range_, (mean_part, var_part) in means_dict.items():
        #                 means, vars = all_new_means_vars[name][range_]
        #                 means[i] = mean_part
        #                 vars[i] = var_part

        for i in range(n):
            i, res = compute_clipped_means_vars(i, ranges=self.requested.all_mean_std_by_percentile_range)
            for name, means_dict in res.items():
                for range_, (mean_part, var_part) in means_dict.items():
                    means, vars = all_new_means_vars[name][range_]
                    means[i] = mean_part
                    vars[i] = var_part

        for name, means_vars in all_new_means_vars.items():
            for range_, (means, vars) in means_vars.items():
                mean = numpy.mean(means, dtype=numpy.float64)
                var = numpy.mean((vars + (means - mean) ** 2))
                self.computed.all_mean_std_by_percentile_range[name][range_] = (float(mean), float(numpy.sqrt(var)))

        computed_dict = {f.name: dict(getattr(self.computed, f.name)) for f in fields(self.computed) if f.init}
        try:
            with self.path.open("w") as file:
                # todo: serialize tuples (range keys)
                yaml.dump(computed_dict, file)
        except Exception as e:
            logger.error(e, exc_info=True)

    def get_percentiles(self, name: str, percentiles: Sequence[float]) -> List[float]:
        ret = [self.computed.all_percentiles[name].get(p, None) for p in percentiles]
        if None in ret:
            # with DatasetStat_rlock:
            # check if meanwhile another thread did the job
            ret = [self.computed.all_percentiles[name].get(p, None) for p in percentiles]

            if None in ret:
                self.request(all_percentiles={name: {p for p, r in zip(percentiles, ret) if r is None}})
                self.compute_requested()
                ret = self.get_percentiles(name, percentiles)

        return ret

    def get_mean_std(self, name: str, percentile_range: Tuple[float, float]) -> Tuple[float, float]:
        ret = self.computed.all_mean_std_by_percentile_range[name].get(percentile_range, None)
        if ret is None:
            # with DatasetStat_rlock:
            # check if meanwhile another thread did the job
            ret = self.computed.all_mean_std_by_percentile_range[name].get(percentile_range, None)
            if ret is None:
                self.request(all_mean_std_by_percentile_range={name: {percentile_range}})
                self.compute_requested()
                ret = self.get_mean_std(name, percentile_range)

        return ret
