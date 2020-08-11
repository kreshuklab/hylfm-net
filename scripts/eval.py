import logging
import os
import typing
from argparse import ArgumentParser
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from pprint import pprint
from typing import Dict

from plain.plain_setup import get_setup

# if __name__ == "__main__":
#     os.environ["OMP_NUM_THREADS"] = "1"
#     os.environ["OPENBLAS_NUM_THREADS"] = "1"
#     os.environ["MKL_NUM_THREADS"] = "1"
#     os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
#     os.environ["NUMEXPR_NUM_THREADS"] = "1"
#     os.nice(19)


from ruamel.yaml import YAML
from tqdm import tqdm

import lnet.transformations
from lnet import get_metric
from lnet.datasets import TensorInfo, ZipDataset, get_dataset_from_info
from lnet.plain_metrics import Metric
from lnet.transformations import ComposedTransformation

logger = logging.getLogger(__name__)

yaml = YAML(typ="safe")


# def slice_along(slice_dim: int, tensors: dict):
#     batch_len = None
#     slice_dim_len = None
#
#     for name, tensor in tensors.items():
#         if isinstance(tensor, list):
#             if batch_len is None:
#                 batch_len = len(tensor)
#             else:
#                 assert batch_len == len(tensor)
#         else:
#             if batch_len is None:
#                 batch_len = tensor.shape[0]
#             else:
#                 assert batch_len == tensor.shape[0]
#
#             assert len(tensor.shape) > slice_dim
#             if slice_dim_len is None:
#                 slice_dim_len = tensor.shape[slice_dim]
#             else:
#                 assert slice_dim_len == tensor.shape[slice_dim], (slice_dim_len, tensor.shape)
#
#     slice_meta_key = f"slice_at_dim_{slice_dim}"
#     for i in range(slice_dim_len):
#         slice_tensors = {
#             k: v if isinstance(v, list) else v[tuple([slice(None)] * slice_dim + [i])] for k, v in tensors.items()
#         }
#         for meta in slice_tensors["meta"]:
#             assert slice_meta_key not in meta, meta
#
#         slice_tensors["meta"] = [{slice_meta_key: i, **meta} for meta in slice_tensors["meta"]]
#
#         yield slice_tensors


def compute_metrics_individually(
    metrics: typing.List[Dict[str, Dict[str, typing.Any]]], tensors: typing.OrderedDict, per_z: bool = False
) -> Dict[str, float]:
    out = {}
    for metrics_group in metrics:
        metrics_group = {name: get_metric(name=name, kwargs=kwargs) for name, kwargs in metrics_group.items()}
        for name, m in metrics_group.items():
            if per_z:
                assert False
                computed = defaultdict(list)
                for zslice_tensors in slice_along(2, tensors):
                    m.update(zslice_tensors)
                    for k, v in m.compute().items():
                        computed[k].append(v)

                    m.reset()
                computed = dict(computed)
            else:
                m.update(tensors)
                computed = m.compute()
                m.reset()

            assert isinstance(computed, dict)
            for k, v in computed.items():
                assert k not in out, (k, list(out.keys()))
                out[k] = v

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
    parser.add_argument("trfs", type=Path)
    parser.add_argument("metrics", type=Path)
    parser.add_argument("--per_z", action="store_true")
    parser.add_argument("--postfix", default="")

    args = parser.parse_args()

    setup = get_setup(subpath=args.subpath)

    print("eval with:")
    pprint(setup)

    model_name: str = setup["model_name"]
    gt_name: str = setup["gt_name"]
    test_data_path: Path = setup["test_data_path"]
    log_path = test_data_path / model_name / f"metrics{args.postfix}"
    log_path.mkdir(exist_ok=True, parents=True)

    datasets = OrderedDict()
    print("get reconstruction from", test_data_path / model_name)
    datasets["pred"] = get_dataset_from_info(
        TensorInfo(
            name="pred",
            root=test_data_path / model_name,
            location=f"*.tif",
            insert_singleton_axes_at=[0, 0],
            remove_singleton_axes_at=[],  #  if care_setup else [-1]
            meta={"log_path": log_path},
        )
    )
    print("get gt from", test_data_path / gt_name)
    datasets[gt_name] = get_dataset_from_info(
        TensorInfo(
            name=gt_name,
            root=test_data_path / gt_name,
            location=f"*.tif",
            insert_singleton_axes_at=[0, 0],
            # remove_singleton_axes_at=[-1],
        )
    )
    trf = ComposedTransformation(
        *[getattr(lnet.transformations, name)(**kwargs) for trf in yaml.load(args.trfs) for name, kwargs in trf.items()]
    )
    ds = ZipDataset(datasets, join_dataset_masks=False, transformation=trf)
    # print(ds[0][setup["pred"]].shape, ds[0][gt_name].shape)

    metrics = yaml.load(args.metrics)
    with ThreadPoolExecutor(max_workers=2) as executor:

        def compute_metrics_individually_from_idx(idx: int):
            return compute_metrics_individually(metrics, ds[idx], per_z=args.per_z)

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

    pprint(list(computed_metrics.keys()))
    # yaml.dump({"metrics": {key: float(val) for key, val in out.metrics.items()}}, log_path / "_summary.yml")
    for name, values in computed_metrics.items():
        yaml.dump(values, log_path / f"{name}.yml")
