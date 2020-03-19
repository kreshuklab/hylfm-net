import logging
from concurrent.futures.thread import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Optional

import numpy
import pandas
import torch
import yaml

LOGS = Path(__file__).parent / "../logs"

logger = logging.getLogger(__name__)


class Collector:
    class valid_names:
        general = {"start time", "config name", "model name", "z_out"}
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
        other = {"#parameters", "test step count", "output scale"}

        def __repr__(self):
            return (
                f"valid names for Collector.get():\n"
                f"general: {self.general}\n"
                f"test_metric: {self.test_metric}\n"
                f"other: {self.other}"
            )

    def __init__(self, glob: str, path: Optional[Path] = None):
        assert glob.endswith(".yml")
        if path is None:
            path = LOGS

        assert path.exists()
        self.spec_paths = [sp for sp in path.glob(glob) if self._is_done(sp) or self._has_at_least_one_epoch(sp)]
        assert self.spec_paths, path / glob
        start_times = [sp.parent.name.split("_") for sp in self.spec_paths]
        start_times = [" ".join(["20" + st[0], st[1].replace("-", ":")]) for st in start_times]
        # start_times = [pandas.Timestamp(datetime.fromisoformat(st)) for st in start_times]
        self.data = pandas.DataFrame(start_times, columns=["start time"])
        config_name = [sp.stem for sp in self.spec_paths]
        self.data.insert(0, "config name", config_name)
        model_name = []
        z_out = []
        for sp in self.spec_paths:
            try:
                with sp.open() as f:
                    config = yaml.safe_load(f)

                model_name.append(config["model"]["name"])
                z_out.append(config["model"]["z_out"])
            except Exception:
                logger.error(sp)
                raise

        self.data.insert(0, "model name", model_name)
        self.data.insert(0, "z_out", z_out)

    def get(self, name: str):
        if name == "trained_models":
            raise ValueError("trained_models, use specific get instead")

        if name not in self.data:
            if name in self.valid_names.general:
                raise NotImplementedError("general metrics should be computed in init!")
            elif name in self.valid_names.test_metric:
                values = self.get_test_metric(name)
            elif name in self.valid_names.other:
                values = self.get_for_each_spec_path(name)
            else:
                raise NotImplementedError

            self.data.insert(0, name, values)

        return self.data[name]

    def get_all(self):
        for metric in self.valid_names.test_metric:
            self.get(metric)

        for metric in self.valid_names.other:
            self.get(metric)

    @staticmethod
    def _is_done(spec: Path):
        test_data_path = spec.parent / "test_data"
        return test_data_path.exists() and any(test_data_path.iterdir())

    @staticmethod
    def _has_at_least_one_epoch(spec: Path):
        models_path = spec.parent / "models"
        return models_path.exists() and any(models_path.iterdir())

    @staticmethod
    def _get_metric_from_test_txt(txt: Path):
        if not txt.exists():
            return float("nan")

        with txt.open() as f:
            line = f.read().strip("\n").split()
            assert len(line) == 2
            return float(line[1])

    def _extract_test_metric(self, name: str):
        return numpy.asarray(
            [self._get_metric_from_test_txt(sp.parent / f"test_data/{name}.txt") for sp in self.spec_paths]
        )

    def get_test_metric(self, name):
        assert name in self.valid_names.test_metric
        return self._extract_test_metric(name.split()[1])

    def get_for_each_spec_path(self, name: str, exc_info: bool = True):
        if name == "output scale":
            get_fn = self._get_output_scale
        elif name == "#parameters":
            get_fn = self._get_nr_parameters
        elif name == "test step count":
            get_fn = self._get_test_step_count
        elif name == "trained_models":
            get_fn = self._get_trained_models
        else:
            raise NotImplementedError(name)

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(get_fn, sp) for sp in self.spec_paths]
            values = []
            for fut in futures:
                try:
                    val = fut.result()
                except Exception as e:
                    logger.error(e, exc_info=exc_info)
                    values.append(numpy.nan)
                else:
                    values.append(val)

        return values

    def _get_test_step_count(self, spec: Path):
        txt = spec.parent / f"test_data/loss.txt"
        if not txt.exists():
            return 0

        with txt.open() as f:
            line = f.read().strip("\n").split()
            assert len(line) == 2
            return int(line[0])

    def _get_nr_parameters(self, spec: Path):
        pth_path = next(spec.parent.glob("models/v0_model_*.pth"))
        with pth_path.open("br") as f:
            model_state = torch.load(f, map_location=torch.device("cpu"))

        nparams = 0
        for name, state in model_state.items():
            nparams += int(numpy.prod(state.shape))

        return nparams

    def _get_output_scale(self, spec: Path):
        with spec.open() as f:
            spec = yaml.safe_load(f)

        default_2d_scales = {"M12": 1, "M13": 1, "M15": 1, "M16": 1, "M17": 1, "M18": 1, "M19": 1, "M20": 1, "D01": 1}
        default_3d_scales = {"M12": 4, "M13": 8, "M15": 4, "M16": 4, "M17": 4, "M18": 4, "M19": 4, "M20": 2, "D01": 8}

        name = spec["model"]["name"]
        kwargs = spec["model"]["kwargs"]
        if name == "M20":
            assert kwargs.get("dilation", 1) == 1

        ignore = [
            "affine_transform_classes",
            "interpolation_order",
            "inplanes_3d",
            "final_activation",
            "dilation",
            "n_res2d_1",
            "kernel_size",
            "stride",
            "padding",
            "output_padding",
            "bias",
            "affine_transform_class",
            "grid_sampling_scale",  # todo: account for when used
            "z_out",
            "growth_rate",
            "bn_size",
            "batch_norm",
            "drop_rate",
        ]

        for ig in ignore:
            kwargs.pop(ig, None)

        scale_2d = None
        scale_3d = None

        n_res2d = kwargs.pop("n_res2d", None)
        if n_res2d is not None:
            assert scale_2d is None
            n_res2d = [n for n in n_res2d if isinstance(n, str)]
            scale_2d = 2 ** "".join(n_res2d).count("u")

        block_config_2d = kwargs.pop("block_config_2d", None)
        if block_config_2d is not None:
            assert scale_2d is None
            block_config_2d = [n for n in block_config_2d if isinstance(n, str)]
            scale_2d = 2 ** "".join(block_config_2d).count("u")

        n_res3d = kwargs.pop("n_res3d", None)
        if n_res3d is not None:
            assert scale_3d is None
            scale_3d = 2 ** [len(n) if isinstance(n, (list, tuple)) else 1 for n in n_res3d].count(2)

        assert not kwargs, kwargs
        if scale_2d is None:
            scale_2d = default_2d_scales[name]

        if scale_3d is None:
            scale_3d = default_3d_scales[name]

        assert scale_2d is not None
        assert scale_3d is not None
        return scale_2d * scale_3d

    def get_trained_models(self):
        trained = self.get_for_each_spec_path("trained_models")
        return {
            sp.resolve().as_posix().replace("/kreshuk.embl.de", "g"): tr
            for sp, tr in zip(self.spec_paths, trained)
            if tr
        }

    def _get_trained_models(self, spec: Path) -> Dict[str, Any]:
        pth_paths = list(spec.parent.glob("models/v0_model_*.pth"))
        if not pth_paths:
            return {}

        pth_epochs = [int(pth.stem.split("_")[-1]) for pth in pth_paths]
        pth_path = pth_paths[pth_epochs.index(min(pth_epochs))].resolve()
        with spec.open() as f:
            spec_dict = yaml.safe_load(f)

        model_dict = spec_dict["model"]
        eval_dict = spec_dict["eval"]
        eval_dict.pop("valid_data", None)
        eval_dict.pop("test_data", None)
        model_dict["checkpoint"] = pth_path.as_posix().replace("/kreshuk.embl.de", "g")

        return {"model": model_dict, "eval": eval_dict}

    def write_out_trained_models(self, category: str):
        out = Path(__file__).parent / f"../trained_models/{category}.yml"
        assert out.parent.exists()
        with out.open("w") as f:
            yaml.safe_dump(self.get_trained_models(), f)

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    collector = Collector("fish/**/*.yml")
    print(len(collector))

    collector.write_out_trained_models("fish")

    # collector.get("output scale")
    # collector.get_all()
    # print(collector.data.head)
    #
    # filtered_data = collector.data.loc[[p is not None for p in collector.data["#parameters"]]]
    # print(filtered_data.head)

    # import plotly.express
    # import matplotlib.pyplot as plt
    # fig = plt.figure(figsize=(10, 10))
    # # seaborn.scatterplot(x, y, data=collector.data)
    # fig = plotly.express.scatter(
    #     filtered_data, x="test step_count", y="test loss", size="#parameters", hover_data=["start time"]
    # )
    #
    # fig.show()
