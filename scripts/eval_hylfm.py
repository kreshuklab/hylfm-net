import logging
import os
import typing
from argparse import ArgumentParser
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from pprint import pprint
from typing import Dict

from ruamel.yaml import YAML
from tqdm import tqdm

import hylfm.transformations
from lnet import get_metric
from hylfm.datasets import TensorInfo, ZipDataset, get_dataset_from_info
from hylfm.plain_metrics import Metric
from hylfm.transformations import ComposedTransformation
from notebooks.care.setup_inference import get_care_setup, get_hylfm_setup

logger = logging.getLogger(__name__)

yaml = YAML(typ="safe")


def compute_metrics_individually(
    metrics: Dict[str, Dict[str, typing.Any]], tensors: typing.OrderedDict
) -> Dict[str, float]:
    metrics = {name: get_metric(name=name, kwargs=kwargs) for name, kwargs in metrics.items()}
    out = {}
    for name, m in metrics.items():
        m.update(tensors)
        computed = m.compute()
        assert isinstance(computed, dict)
        for k, v in computed.items():
            assert k not in out, (k, list(out.keys()))
            out[k] = v

        m.reset()

    return out


def update_metrics(metrics: Dict[str, Metric], tensors: typing.OrderedDict) -> None:
    for name, m in metrics.items():
        m.update(tensors)


def compute_and_reset_metrics(metrics: Dict[str, Metric]) -> Dict[str, float]:
    out = {}
    for name, m in metrics.items():
        computed = m.compute()
        assert isinstance(computed, dict)
        for k, v in computed.items():
            assert k not in out, (k, list(out.keys()))
            out[k] = v

        m.reset()

    return out


if __name__ == "__main__":
    # os.environ["OMP_NUM_THREADS"] = "4" # export OMP_NUM_THREADS=4
    # os.environ["OPENBLAS_NUM_THREADS"] = "4" # export OPENBLAS_NUM_THREADS=4
    # os.environ["MKL_NUM_THREADS"] = "6" # export MKL_NUM_THREADS=6
    # os.environ["VECLIB_MAXIMUM_THREADS"] = "4" # export VECLIB_MAXIMUM_THREADS=4
    # os.environ["NUMEXPR_NUM_THREADS"] = "6" # export NUMEXPR_NUM_THREADS=6
    os.nice(20)
    parser = ArgumentParser(description="eval hylfm")
    parser.add_argument("model_name")
    parser.add_argument("trfs", type=Path)
    parser.add_argument("metrics", type=Path)

    args = parser.parse_args()

    setup = get_hylfm_setup(model_name=args.model_name)
    print("eval with:")
    pprint(setup)

    result_path: Path = setup["result_path"]
    assert result_path.exists(), result_path.absolute()
    datasets = OrderedDict()
    log_path = result_path / "metrics"
    log_path.mkdir(exist_ok=True)
    datasets[setup["pred"]] = get_dataset_from_info(
        TensorInfo(
            name=setup["pred"],
            root=setup["model_basedir"] / setup["subpath"],
            location=f"*.tif",
            insert_singleton_axes_at=[0, 0],
            remove_singleton_axes_at=[-1],
            meta={"log_dir": log_path},
        )
    )
    datasets[setup["gt"]] = get_dataset_from_info(
        TensorInfo(
            name=setup["gt"],
            root=setup["gt_path"],
            location=f"*.tif",
            insert_singleton_axes_at=[0, 0],
            # remove_singleton_axes_at=[-1],
        )
    )
    print("get trfs from", args.trfs.absolute())
    assert args.trfs.exists()
    trf = ComposedTransformation(
        *[
            getattr(hylfm.transformations, name)(**kwargs)
            for trf in yaml.load(args.trfs)
            for name, kwargs in trf.items()
        ]
    )
    ds = ZipDataset(datasets, join_dataset_masks=False, transformation=trf)
    # print(ds[0][setup["pred"]].shape, ds[0][gt_name].shape)

    metrics = yaml.load(args.metrics)
    with ThreadPoolExecutor(max_workers=2) as executor:

        def compute_metrics_individually_from_idx(idx: int):
            return compute_metrics_individually(metrics, ds[idx])

        futs = [executor.submit(compute_metrics_individually_from_idx, idx) for idx in range(len(ds))]

        for fut in tqdm(as_completed(futs), total=len(futs)):
            e = fut.exception()
            if e is not None:
                raise e

            assert e is None

        computed_metrics = [fut.result() for fut in futs]

    # # metrics_for_epoch = {name: get_metric(name=name, kwargs=kwargs) for name, kwargs in metrics.items()}
    # dataloader = DataLoader(ds, num_workers=2, collate_fn=collate_fn)
    # computed_metrics = []
    # for tensors in dataloader:
    #     computed_metrics.append(compute_metrics_individually(metrics, tensors))

    computed_metrics = {name: [cm[name] for cm in computed_metrics] for name in computed_metrics[0]}

    pprint(computed_metrics)
    # yaml.dump({"metrics": {key: float(val) for key, val in out.metrics.items()}}, log_dir / "_summary.yml")
    for name, values in computed_metrics.items():
        yaml.dump(values, log_path / f"{name}.yml")
