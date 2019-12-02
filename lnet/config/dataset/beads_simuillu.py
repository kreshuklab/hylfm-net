from pathlib import Path

from lnet.config.dataset import NamedDatasetInfo

beads_01mu_0_simuillu = {
        f"beads_01mu_0_simuillu_{i}": NamedDatasetInfo(
            Path(
                "/g/hufnagel/LF/LenseLeNet_Microscope/20191030_Beads_massiveGT/Beads_1micron/2019-10-30_05.20.28_withSimultIllum"
            ),
            f"stack_1_channel_2/TP_*/RC_rectified/Cam_Right_{i}_rectified.tif",
            "stack_1_channel_1/TP_*/LC/Cam_Left_registered.tif",
            description=f"beads_01mu_0_simuillu_{i}",
        )
        for i in range(1, 242)
    }

beads_01mu_1_simuillu = {
        f"beads_01mu_1_simuillu_{i}": NamedDatasetInfo(
            Path(
                "/g/hufnagel/LF/LenseLeNet_Microscope/20191030_Beads_massiveGT/Beads_1micron/2019-10-30_05.54.12_withSimultIllum"
            ),
            f"stack_1_channel_2/TP_*/RC_rectified/Cam_Right_{i}_rectified.tif",
            "stack_1_channel_1/TP_*/LC/Cam_Left_registered.tif",
            description=f"beads_01mu_1_simuillu_{i}",
        )
        for i in range(1, 242)
    }

__all__ = [*beads_01mu_0_simuillu, *beads_01mu_1_simuillu]


def __getattr__(name: str) -> NamedDatasetInfo:
    if name in beads_01mu_0_simuillu:
        return beads_01mu_0_simuillu[name]
    elif name in beads_01mu_1_simuillu:
        return beads_01mu_1_simuillu[name]
    else:
        raise AttributeError(name)
