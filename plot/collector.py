import logging
from pathlib import Path
from typing import Optional

import numpy
import pandas
import torch

LOGS = Path(__file__).parent / "../logs"

logger = logging.getLogger(__name__)


class Collector:
    def cached(self, get_fn):
        def cached_get_fn(name: str):
            if name not in self.data:
                self.data.insert(0, name, get_fn())

            return self.data[name]

        return cached_get_fn

    class valid_names:
        general = {"start time", "config name"}
        test_metric = {
            "test bead-precision",
            "test bead-precision",
            "test bead-recall",
            "test loss",
            "test ms-ssim",
            "test nrmse",
            "test psnr",
            "test ssim",
        }
        other = {"#parameters", "test step_count"}

    def __init__(self, glob: str, path: Optional[Path] = None):
        assert glob.endswith(".yml")
        if path is None:
            path = LOGS

        assert path.exists()
        self.spec_paths = [sp for sp in path.glob(glob) if self._guess_done(sp)]
        start_times = [sp.parent.name.split("_") for sp in self.spec_paths]
        start_times = [" ".join(["20" + st[0], st[1].replace("-", ":")]) for st in start_times]
        # start_times = [pandas.Timestamp(datetime.fromisoformat(st)) for st in start_times]
        self.data = pandas.DataFrame(start_times, columns=["start time"])
        config_name = [sp.stem for sp in self.spec_paths]
        self.data.insert(0, "config name", config_name)

    @staticmethod
    def _guess_done(spec: Path):
        test_data_path = spec.parent / "test_data"
        return test_data_path.exists() and any(test_data_path.iterdir())

    def get_test_step_count(self):
        name = "test step_count"
        if name not in self.data:

            def _get_step_from_test_txt(txt: Path):
                if not txt.exists():
                    return 0

                with txt.open() as f:
                    line = f.read().strip("\n").split()
                    assert len(line) == 2
                    return int(line[0])

            self.data.insert(
                0, name, [_get_step_from_test_txt(sp.parent / f"test_data/loss.txt") for sp in self.spec_paths]
            )

        return self.data[name]

    @staticmethod
    def _get_metric_from_test_txt(txt: Path):
        if not txt.exists():
            return float("nan")

        with txt.open() as f:
            line = f.read().strip("\n").split("\t")
            assert len(line) == 2
            return float(line[1])

    def _extract_test_metric(self, name: str):
        return numpy.asarray(
            [self._get_metric_from_test_txt(sp.parent / f"test_data/{name}.txt") for sp in self.spec_paths]
        )

    def get_test_metric(self, name):
        assert name in self.valid_names.test_metric
        if name not in self.data:
            value = self._extract_test_metric(name.split()[1])
            self.data.insert(0, name, value)

        return self.data[name]

    def _get_nr_parameters(self, pth_path: Path):
        with pth_path.open("br") as f:
            model_state = torch.load(f, map_location=torch.device("cpu"))

        nparams = 0
        for name, state in model_state.items():
            nparams += int(numpy.prod(state.shape))

        return nparams

    def get_nr_parameters(self):
        name = "#parameters"
        assert name in self.valid_names.other
        if name not in self.data:
            values = []
            for sp in self.spec_paths:
                try:
                    val = self._get_nr_parameters(next(sp.parent.glob("models/v0_model_*.pth")))
                except Exception as e:
                    logger.error(e)
                    print(e)
                    values.append(numpy.nan)
                else:
                    values.append(val)

            self.data.insert(0, name, values)

        return self.data[name]

    def get(self, name: str):
        if name in self.valid_names.general:
            return self.data[name]
        elif name in self.valid_names.test_metric:
            return self.get_test_metric(name)
        elif name in self.valid_names.other:
            if name == "test step_count":
                return self.get_test_step_count()
            elif name == "#parameters":
                return self.get_nr_parameters()
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError


if __name__ == "__main__":
    import plotly.express
    import matplotlib.pyplot as plt

    collector = Collector("fish/**/*.yml")

    x = "test step_count"
    y = "test loss"
    size = "#parameters"
    hover_data = ["start time"]
    collector.get(x)
    collector.get(y)
    collector.get(size)
    [collector.get(hd) for hd in hover_data]
    print(collector.data.head)

    filtered_data = collector.data.loc[[p is not None for p in collector.data["#parameters"]]]
    print(filtered_data.head)

    fig = plt.figure(figsize=(10, 10))
    # seaborn.scatterplot(x, y, data=collector.data)
    fig = plotly.express.scatter(filtered_data, x=x, y=y, size=size, hover_data=hover_data)

    fig.show()
