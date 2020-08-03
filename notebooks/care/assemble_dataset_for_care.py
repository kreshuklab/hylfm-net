import tifffile
from argparse import ArgumentParser
from concurrent.futures.thread import ThreadPoolExecutor
from pathlib import Path

from lnet.setup.base import DatasetGroupSetup
from lnet.datasets import get_collate_fn

from ruamel.yaml import YAML

yaml = YAML(typ="safe")

def volwrite(p: Path, data, compress=2, **kwargs):
    with p.open("wb") as f:
        tifffile.imsave(f, data, compress=compress, **kwargs)
    
def assemble_dataset_for_care(config: Dict[str, Any], data_path: Path, x="lr", y="ls_trf"):
    group_setup = DatasetGroupSetup(batch_size=1, **config)
    dataset = group_setup.dataset
#     data_loader = torch.utils.data.DataLoader(
#                 dataset=group_setup.dataset,
#                 shuffle=False,
#                 collate_fn=get_collate_fn(lambda batch: batch),
#                 num_workers=16,
#                 pin_memory=False,
#             )

    def save_image_pair(idx):
        tensors = dataset[idx]
        x = tensors[x]
        y = tensors[y]
        assert isinstance(x, numpy.ndarray)
        assert isinstance(y, numpy.ndarray)
        assert x.shape == y.shape
        assert x.dtype == numpy.uint16
        assert y.dtype == numpy.uint16
        
        volwrite(data_path / "x", x)
        volwrite(data_path / "y", y)

    with ThreadPoolExecutor(max_workers=1) as executor:
        futs = []
        for idx in indices:
            futs.append(executor.submit(save_image_pair, idx))

        for fut in as_completed(futs):
            exc = fut.exception()
            if exc is not None:
                raise exc

if __name__ == "__main__":
    parser = ArgumentParser(description="care prepper")
    parser.add_argument("config_path", type=Path)
    parser.add_argument("names", nargs="?")
    parser.add_argument("data_root", type=Path, default=Path("/scratch/beuttenm/lnet/care"))
                 
    args = parser.parse_args()
    assert args.config_path.exists()
    print("config_path:", args.config_path)
    data_path = args.data_path
    config = yaml.load(args.config_path)
    print("selecting", args.names)
    for name in args.names:
        config = config[name]
        data_path /= name
                 
    print("save to", data_path)
    assemble_dataset_for_care(config, data_path=data_path)
