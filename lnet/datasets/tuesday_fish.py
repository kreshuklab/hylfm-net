from pathlib import Path

from .base import NamedDatasetInfo

tuesday_fish0 = {
    f"tuesday_fish0_{i:05}": NamedDatasetInfo(
        Path(
            f"/g/hufnagel/LF/LenseLeNet_Microscope/20191202_staticHeart_dynamicHeart/data/2019-12-02_04.12.36_10msExp/stack_1_channel_3/TP_{i:05}/RC_rectified"
        ),
        "Cam_Right_*_rectified.tif",
        description=f"tuesday_fish0_{i:05}",
    )
    for i in range(40)
}

__all__ = [*tuesday_fish0]


def __getattr__(name: str) -> NamedDatasetInfo:
    if name in tuesday_fish0:
        return tuesday_fish0[name]
    else:
        raise AttributeError(name)
