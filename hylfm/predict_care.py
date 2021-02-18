import logging
from pathlib import Path

from tifffile import imread
from tqdm import tqdm

from hylfm.utils.io import save_tensor

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import typer

logger = logging.getLogger(__name__)


app = typer.Typer()


@app.command(name="test_care")
def tst_care(care_model_path: Path, lfd_path: Path):
    assert lfd_path.name == "pred"
    assert lfd_path.parent.name == "lfd"
    from csbdeep.io import save_tiff_imagej_compatible
    from csbdeep.models import CARE

    axes = "ZYX"
    model = CARE(config=None, name=care_model_path.name, basedir=care_model_path.parent)
    care_result_root = lfd_path.parent.parent / "care" / "pred"
    care_result_root.mkdir(parents=True, exist_ok=True)
    print("saving CARE reconstructions to", care_result_root)

    for file_path in tqdm(list(lfd_path.glob("*tif"))):
        result_path = care_result_root / file_path.name
        if result_path.exists():
            continue

        x = imread(str(file_path)).squeeze()
        restored = model.predict(x, axes)
        restored = restored.squeeze()[None, ...]
        save_tensor(result_path, restored)
        # save_tiff_imagej_compatible(str(result_path), restored, axes)


if __name__ == "__main__":
    app()
