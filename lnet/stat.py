from collections import defaultdict
from concurrent.futures import as_completed, ThreadPoolExecutor
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Optional, Tuple, Set, DefaultDict, List, Sequence

import numpy
import yaml

from torch.utils.data import Dataset


@dataclass
class ComputedDatasetStat:
    all_percentiles: DefaultDict[int, Dict[float, float]] = field(default_factory=lambda: defaultdict(dict))
    all_mean_std_by_percentile_range: DefaultDict[int, Dict[Tuple[float, float], Tuple[float, float]]] = field(
        default_factory=lambda: defaultdict(dict)
    )


@dataclass
class RequestedDatasetStat:
    all_percentiles: DefaultDict[int, Set[float]] = field(default_factory=lambda: defaultdict(set))
    all_mean_std_by_percentile_range: DefaultDict[int, Set[Tuple[float, float]]] = field(
        default_factory=lambda: defaultdict(set)
    )


class DatasetStat:
    computed: Optional[ComputedDatasetStat]
    requested: RequestedDatasetStat

    def __init__(
        self,
        path: Path,
        dataset: Dataset,
        percentiles: Optional[Dict[int, Set[float]]],
        means: Optional[Dict[int, Set[Tuple[float, float]]]],
    ):
        assert path.suffix == ".yml", path.suffix == "yaml"
        self.path = path
        self.dataset = dataset
        if path.exists():
            with path.open() as f:
                self.computed = ComputedDatasetStat(**yaml.safe_load(f))

        else:
            self.computed = None

        if percentiles is None:
            percentiles = {}

        if means is None:
            means = {}

        self.requested = RequestedDatasetStat(
            all_percentiles=defaultdict(set, **percentiles), all_mean_std_by_percentile_range=defaultdict(set, **means)
        )

    def request(
        self,
        all_percentiles: Optional[Dict[int, Set[float]]] = None,
        all_mean_std_by_percentile_range: Optional[Dict[int, Set[Tuple[float, float]]]] = None,
    ):
        if all_mean_std_by_percentile_range:
            for idx, pranges in all_mean_std_by_percentile_range.items():
                self.requested.all_mean_std_by_percentile_range[idx] |= pranges - set(
                    self.computed.all_mean_std_by_percentile_range[idx]
                )

                # also request all involved percentiles
                req = set().union(*{set(pr) for pr in pranges})
                self.requested.all_percentiles[idx] |= req - set(self.computed.all_percentiles[idx])

        if all_percentiles:
            for idx, pers in all_percentiles.items():
                self.requested.all_percentiles[idx] |= pers - set(self.computed.all_percentiles[idx])

    def compute_requested(self):
        n = len(self.dataset)
        sample = self.dataset[0]
        nbins = numpy.iinfo(numpy.uint16).max // 5
        hist_min = 0.0
        hist_max = numpy.iinfo(numpy.uint16).max
        bin_width = (hist_max - hist_min) / nbins

        hist_path = self.path.with_suffix(".hist.npy")
        if hist_path.exists():
            hist = numpy.load(hist_path)
        else:
            hist = numpy.zeros((len(sample), nbins), numpy.uint64)

            def compute_hist(i: int):
                return numpy.stack(
                    [numpy.histogram(s, bins=nbins, range=(hist_min, hist_max))[0] for s in self.dataset[i]]
                )

            futs = []
            with ThreadPoolExecutor(max_workers=16) as executor:
                for i in range(n):
                    futs.append(executor.submit(compute_hist, i))

                for fut in as_completed(futs):
                    e = fut.exception()
                    if e is not None:
                        raise e

                    hist += fut.result()

            numpy.save(hist_path, hist)

        cumsum = hist.cumsum(axis=1)

        all_new_percentiles: Dict[int, Dict[float, float]] = {}
        for idx, pers in self.requested.all_percentiles.items():
            pers = numpy.array(pers)
            per_values = numpy.searchsorted(cumsum[idx], pers * cumsum[idx, -1] / 100) * bin_width
            all_new_percentiles[idx] = dict(zip(pers.tolist(), per_values.tolist()))

        self.computed = self.computed or ComputedDatasetStat()
        for idx, new_percentiles in all_new_percentiles.items():
            self.computed.all_percentiles[idx].update(new_percentiles)

        def compute_clipped_means_vars(
            i: int, ranges: DefaultDict[int, Set[Tuple[float, float]]]
        ) -> DefaultDict[int, Dict[Tuple[float, float, Tuple[float, float]]]]:
            ret = defaultdict(dict)
            for idx, data in enumerate(self.dataset[i]):
                for range_ in ranges[idx]:
                    lower, upper = range_
                    if lower is not None or upper is not None:
                        data = numpy.clip(
                            data,
                            a_min=None if lower is None else self.computed.all_percentiles[idx][lower],
                            a_max=None if upper is None else self.computed.all_percentiles[idx][upper],
                        )

                    ret[idx][range_] = (
                        numpy.mean(data, dtype=numpy.float64).item(),
                        numpy.var(data, dtype=numpy.float64).item(),
                    )

            return ret

        all_new_means_vars: DefaultDict[
            int, DefaultDict[Tuple[float, float]], Tuple[numpy.array, numpy.array]
        ] = defaultdict(
            lambda: defaultdict((numpy.empty((n,), dtype=numpy.float64), numpy.empty((n,), dtype=numpy.float64)))
        )
        with ThreadPoolExecutor(max_workers=16) as executor:
            futs = [
                executor.submit(compute_clipped_means_vars, i=i, ranges=self.requested.all_mean_std_by_percentile_range)
                for i in range(n)
            ]

            for fut in as_completed(futs):
                e = fut.exception()
                if e is not None:
                    raise e

                fut_res: Tuple[int, DefaultDict[int, Dict[Tuple[float, float, Tuple[float, float]]]]] = fut.result()
                i, res = fut_res
                for idx, means_dict in res.items():
                    for range_, (mean_part, var_part) in means_dict.items():
                        means, vars = all_new_means_vars[idx][range_]
                        means[i] = mean_part
                        vars[i] = var_part

        for idx, means_vars in all_new_means_vars:
            for range_, (means, vars) in means_vars.items():
                mean = numpy.mean(means, dtype=numpy.float64)
                var = numpy.mean((vars + (means - mean) ** 2))
                self.computed.all_mean_std_by_percentile_range[idx][range_] = (float(mean), float(numpy.sqrt(var)))

        with self.path.open("w") as f:
            yaml.safe_dump(asdict(self.computed), f)

    def get_percentiles(self, idx: int, percentiles: Sequence[float]) -> List[float]:
        if self.computed is None:
            ret = [None] * len(percentiles)
        else:
            ret = [self.computed.all_percentiles[idx].get(p, None) for p in percentiles]

        if None in ret:
            self.request(percentiles={idx: {p for p, r in zip(percentiles, ret) if r is None}})
            self.compute_requested()
            return self.get_percentiles(idx, percentiles)
        else:
            return ret

    def get_mean_std(self, idx: int, percentile_range: Tuple[float, float]) -> Tuple[float, float]:
        mean_std = (
            None
            if self.computed is None
            else self.computed.all_mean_std_by_percentile_range[idx].get(percentile_range, None)
        )
        if mean_std is None:
            self.request(means={idx: {percentile_range}})
            self.compute_requested()
            return self.get_mean_std(idx, percentile_range)
        else:
            return mean_std
