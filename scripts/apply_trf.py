import os
from argparse import ArgumentParser
from collections import OrderedDict
from concurrent.futures import as_completed, ThreadPoolExecutor
from pathlib import Path

from ruamel.yaml import YAML
from tifffile import imsave
from tqdm import tqdm

from hylfm.datasets import TensorInfo, ZipDataset, get_dataset_from_info, get_tensor_info
from hylfm.transformations.utils import get_composed_transformation_from_config

yaml = YAML(typ="safe")


if __name__ == "__main__":
    os.nice(10)
    parser = ArgumentParser(description="care inference")
    parser.add_argument("subpath", default="heart/dynamic")
    parser.add_argument("model_name", default="v0_spe1000_on_48x88x88")
    parser.add_argument("trf_config_path", type=Path)
    # parser.add_argument("--file_paths_path", type=Path, default="")
    parser.add_argument("--cuda", default="")

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda

    subpath = args.subpath
    model_name = args.model_name

    meta = {
        "z_out": 49,
        "nnum": 19,
        "interpolation_order": 2,
        "scale": 4,
        "z_ls_rescaled": 241,
        "pred_z_min": 0,
        "pred_z_max": 838,
    }
    datasets = OrderedDict()
    datasets["pred"] = get_dataset_from_info(
        TensorInfo(
            name="pred",
            root=Path("/scratch/beuttenm/lnet/care/results"),
            location=f"{subpath}/{model_name}/*.tif",
            insert_singleton_axes_at=[0, 0],
            z_slice=None,
            meta={"crop_name": "Heart_tightCrop", **meta},
        ),
        cache=True,
    )
    datasets["ls_slice"] = get_dataset_from_info(
        get_tensor_info("heart_dynamic.2019-12-09_04.54.38", name="ls_slice", meta=meta),
        cache=True,
        filters=[("z_range", {})],
    )

    assert len(datasets["pred"]) == 51 * 241, len(datasets["pred"])
    assert len(datasets["ls_slice"]) == 51 * 209, len(datasets["ls_slice"])
    # ipt_paths = {
    #     "pred": ,
    #     "ls_slice": Path(
    #         "/g/kreshuk/LF_computed/lnet/logs/heart2/test_z_out49/lr_f4/heart_dynamic.2019-12-09_04.54.38/run000/ds0-0/ls_slice"
    #     ),
    # }
    ds = ZipDataset(datasets)
    assert len(ds) == 51 * 209, len(ds)

    out_paths = {
        "pred_slice": Path("/g/kreshuk/LF_computed/lnet/care/results") / subpath / model_name,
        # "pred_vol": Path("/g/kreshuk/LF_computed/lnet/care/results") / subpath / model_name / "vol",
    }
    for name, p in out_paths.items():
        p.mkdir(parents=True, exist_ok=True)
        print(name, p)

    trf = get_composed_transformation_from_config(yaml.load(args.trf_config_path))

    def do_work(i):
        out_file_paths = {name: p / f"{i:05}.tif" for name, p in out_paths.items()}
        # if all(p.exists() for p in out_file_paths.values()):
        #     return

        sample = ds[i]
        # ipt_tensors = {name: imread(str(p / file_name)) for name, p in ipt_paths.items()}
        # ipt_tensors["meta"] = [{"pred": {"crop_name": "Heart_tightCrop"}}]
        # print('sample keys', list(sample.keys()))
        out_tensors = trf(sample)
        # print(list(out.keys()))
        for name, p in out_file_paths.items():
            out = out_tensors[name]
            imsave(str(p), out)

    with ThreadPoolExecutor(24) as executor:
        futs = [executor.submit(do_work, i) for i in range(len(ds))]

        for fut in tqdm(as_completed(futs), total=len(ds)):
            e = fut.exception()
            if e is not None:
                raise e

            assert e is None
