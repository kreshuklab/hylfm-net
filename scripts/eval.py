import logging
import typing
from argparse import ArgumentParser
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from pathlib import Path
from pprint import pprint
from typing import Dict

from ruamel.yaml import YAML
from torch.utils.data import DataLoader

import lnet.transformations
from lnet import get_metric
from lnet.datasets import TensorInfo, ZipDataset, get_dataset_from_info
from lnet.datasets.base import collate_fn
from lnet.plain_metrics import Metric
from lnet.transformations import ComposedTransformation

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
    parser = ArgumentParser(description="eval")
    parser.add_argument("subpath")
    parser.add_argument("model_name")
    parser.add_argument("trfs", type=Path)
    parser.add_argument("metrics", type=Path)
    parser.add_argument("--pred_root", type=Path, default=Path("/g/kreshuk/LF_computed/lnet/care/results"))
    parser.add_argument("--gt_root", type=Path, default=Path("/g/kreshuk/LF_computed/lnet/care/gt"))

    args = parser.parse_args()

    pred_root: Path = args.pred_root
    assert pred_root.exists(), pred_root.absolute()
    assert (pred_root / args.subpath).exists(), (pred_root / args.subpath).absolute()
    assert (pred_root / args.subpath / args.model_name).exists(), (
        pred_root / args.subpath / args.model_name
    ).absolute()

    gt_name = "ls_reg" if "beads" in args.subpath else "ls_trf"
    gt_root: Path = args.gt_root
    assert gt_root.exists(), gt_root.absolute()
    assert (gt_root / args.subpath).exists(), (gt_root / args.subpath).absolute()
    assert (gt_root / args.subpath / "test").exists(), (gt_root / args.subpath / "test").absolute()
    assert (gt_root / args.subpath / "test" / gt_name).exists(), (gt_root / args.subpath / "test" / gt_name).absolute()

    datasets = OrderedDict()
    log_path = pred_root / args.subpath / args.model_name / "metrics"
    log_path.mkdir(exist_ok=True)
    datasets["pred"] = get_dataset_from_info(
        TensorInfo(
            name="pred",
            root=pred_root,
            location=f"{args.subpath}/{args.model_name}/*.tif",
            insert_singleton_axes_at=[0, 0],
            meta={"log_path": log_path},
        )
    )
    datasets[gt_name] = get_dataset_from_info(
        TensorInfo(
            name=gt_name,
            root=args.gt_root,
            location=f"{args.subpath}/test/{gt_name}/*.tif",
            insert_singleton_axes_at=[0, 0],
        )
    )
    trf = ComposedTransformation(
        *[getattr(lnet.transformations, name)(**kwargs) for trf in yaml.load(args.trfs) for name, kwargs in trf.items()]
    )
    ds = ZipDataset(datasets, join_dataset_masks=False, transformation=trf)
    # print(ds[0]["pred"].shape, ds[0][gt_name].shape)

    metrics = yaml.load(args.metrics)
    with ThreadPoolExecutor(max_workers=32) as executor:

        def compute_metrics_individually_from_idx(idx: int):
            return compute_metrics_individually(metrics, ds[idx])

        futs = [executor.submit(compute_metrics_individually_from_idx, idx) for idx in range(len(ds))]
        computed_metrics = [fut.result() for fut in futs]

    # # metrics_for_epoch = {name: get_metric(name=name, kwargs=kwargs) for name, kwargs in metrics.items()}
    # dataloader = DataLoader(ds, num_workers=2, collate_fn=collate_fn)
    # computed_metrics = []
    # for tensors in dataloader:
    #     computed_metrics.append(compute_metrics_individually(metrics, tensors))

    computed_metrics = {name: [cm[name] for cm in computed_metrics] for name in computed_metrics[0]}

    pprint(computed_metrics)
    # yaml.dump({"metrics": {key: float(val) for key, val in out.metrics.items()}}, log_path / "_summary.yml")
    for name, values in computed_metrics.items():
        yaml.dump(values, log_path / f"{name}.yml")
