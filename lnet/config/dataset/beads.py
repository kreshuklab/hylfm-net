from pathlib import Path

from lnet.config.dataset import NamedDatasetInfo


bead00_LS0 = NamedDatasetInfo(
    Path("/g/hufnagel/LF/LenseLeNet_Microscope/20190531_MassiveGT_GenerationBeads_crop"),
    "lfimg",
    "gt/LS_Stack",
    "bead00_0_LS_561nm",
)

bead00_LF0 = NamedDatasetInfo(
    Path("/g/hufnagel/LF/LenseLeNet_Microscope/20190531_MassiveGT_GenerationBeads_crop"),
    "lfimg",
    "gt/LF_Recon",
    "bead00_0_Recon_561nm",
)
bead00_LS1 = NamedDatasetInfo(
    Path("/g/hufnagel/LF/LenseLeNet_Microscope/20190611_Beads_Series"), "lfimg", "gt/LS_Stack", "BeadsSeriesLS_561nm"
)
bead00_LF1 = NamedDatasetInfo(
    Path("/g/hufnagel/LF/LenseLeNet_Microscope/20190611_Beads_Series"), "lfimg", "gt/LF_Stack", "BeadsSeriesLF_561nm"
)


bead00_simLFfromLS0 = NamedDatasetInfo(
    Path("/g/hufnagel/LF/LenseLeNet_Microscope/20190531_MassiveGT_GenerationBeads_crop_GT_LFproj"),
    "GT_LFproj",
    "gt/LS_Stack",
)

bead00_LS0_35 = NamedDatasetInfo(
    Path("/g/hufnagel/LF/LenseLeNet_Microscope/20190531_MassiveGT_GenerationBeads_crop"),
    "lfimg",
    "gt/LS_Stack",
    "bead00_0_LS_561nm",
    y_roi=(slice(7, 42), slice(None), slice(None)),
)

bead00_LF0_35 = NamedDatasetInfo(
    Path("/g/hufnagel/LF/LenseLeNet_Microscope/20190531_MassiveGT_GenerationBeads_crop"),
    "lfimg",
    "gt/LF_Recon",
    "bead00_0_Recon_561nm",
    y_roi=(slice(7, 42), slice(None), slice(None)),
)


bead00_LS_single_planes_max0 = NamedDatasetInfo(
    Path("/g/hufnagel/LF/LenseLeNet_Microscope/20190531_MassiveGT_GenerationBeads_crop"),
    "LS_IlluminationStacked/RectifiedInput/MAX",
    "gt/LS_Stack",
    "bead00_0_LS_561nm",
)

bead00_LS_single_planes_sum0 = NamedDatasetInfo(
    Path("/g/hufnagel/LF/LenseLeNet_Microscope/20190531_MassiveGT_GenerationBeads_crop"),
    "LS_IlluminationStacked/RectifiedInput/SUM",
    "gt/LS_Stack",
    "bead00_0_LS_561nm",
)


bead01_0 = NamedDatasetInfo(
    Path("/g/hufnagel/DeepLearningReconstruction/NewArtificialBeads49planesVolumes"), "lf_sim", "truth"
)


bead02_test0 = NamedDatasetInfo(
    Path("/g/hufnagel/LF/LenseLeNet_Microscope/20191024_Beam_Beads/Beads/TrainingData/2019-10-24_03.52.50_afterThis_HomogeneousLFillum/stack_0_channel_0/TP_00000"), "RC_rectified",
    description="bead02_test0"
)