from pathlib import Path

from .base import NamedDatasetInfo, GKRESHUK, GHUFNAGELLFLenseLeNet_Microscope

bead00_LS0 = NamedDatasetInfo(
    Path(GHUFNAGELLFLenseLeNet_Microscope) / "/20190531_MassiveGT_GenerationBeads_crop",
    "lfimg",
    "gt/LS_Stack",
    "bead00_0_LS_561nm",
)

bead00_LF0 = NamedDatasetInfo(
    Path(GHUFNAGELLFLenseLeNet_Microscope) / "/20190531_MassiveGT_GenerationBeads_crop",
    "lfimg",
    "gt/LF_Recon",
    "bead00_0_Recon_561nm",
)
bead00_LS1 = NamedDatasetInfo(
    Path(GHUFNAGELLFLenseLeNet_Microscope) / "/20190611_Beads_Series", "lfimg", "gt/LS_Stack", "BeadsSeriesLS_561n"
)
bead00_LF1 = NamedDatasetInfo(
    Path(GHUFNAGELLFLenseLeNet_Microscope) / "/20190611_Beads_Series", "lfimg", "gt/LF_Stack", "BeadsSeriesLF_561n"
)


bead00_simLFfromLS0 = NamedDatasetInfo(
    Path(GHUFNAGELLFLenseLeNet_Microscope) / "/20190531_MassiveGT_GenerationBeads_crop_GT_LFproj",
    "GT_LFproj",
    "gt/LS_Stack",
)

bead00_LS0_35 = NamedDatasetInfo(
    Path(GHUFNAGELLFLenseLeNet_Microscope) / "/20190531_MassiveGT_GenerationBeads_crop",
    "lfimg",
    "gt/LS_Stack",
    "bead00_0_LS_561nm",
    y_roi=(slice(7, 42), slice(None), slice(None)),
)

bead00_LF0_35 = NamedDatasetInfo(
    Path(GHUFNAGELLFLenseLeNet_Microscope) / "/20190531_MassiveGT_GenerationBeads_crop",
    "lfimg",
    "gt/LF_Recon",
    "bead00_0_Recon_561nm",
    y_roi=(slice(7, 42), slice(None), slice(None)),
)


bead00_LS_single_planes_max0 = NamedDatasetInfo(
    Path(GHUFNAGELLFLenseLeNet_Microscope) / "/20190531_MassiveGT_GenerationBeads_crop",
    "LS_IlluminationStacked/RectifiedInput/MAX",
    "gt/LS_Stack",
    "bead00_0_LS_561nm",
)

bead00_LS_single_planes_sum0 = NamedDatasetInfo(
    Path(GHUFNAGELLFLenseLeNet_Microscope) / "/20190531_MassiveGT_GenerationBeads_crop",
    "LS_IlluminationStacked/RectifiedInput/SUM",
    "gt/LS_Stack",
    "bead00_0_LS_561nm",
)


bead01_0 = NamedDatasetInfo(
    Path("/g/hufnagel/DeepLearningReconstruction/NewArtificialBeads49planesVolumes"), "lf_sim", "truth"
)


bead02_test0 = NamedDatasetInfo(
    Path(""),
    GKRESHUK + "/beuttenm/tmp/Cam_Right_rectified*.tif",
    GHUFNAGELLFLenseLeNet_Microscope
    + "/20191024_Beam_Beads/Beads/TrainingData/2019-10-24_03.52.50_afterThis_HomogeneousLFillum/stack_1_channel_2/TP_00000/LC/Cam_Left_registered*.tif",
    description="bead02_test0",
)

bead03_test0 = NamedDatasetInfo(
    Path(GHUFNAGELLFLenseLeNet_Microscope) / "/20191030_Beads_massiveGT/Beads_1micron/2019-10-30_07.23.55",
    "stack_0_channel_0/TP_00000/RC_rectified",
    "stack_1_channel_1/TP_00000/LC/Registration_fromTemplateXML",
    description="bead03_test0",
)

beads_11mu_0 = NamedDatasetInfo(
    Path(GHUFNAGELLFLenseLeNet_Microscope) / "/20191030_Beads_massiveGT/Beads_11micron/2019-10-30_09.37.45",
    "stack_0_channel_0/TP_*/RC_rectified/Cam_Right_1_rectified.tif",
    "stack_1_channel_1/TP_*/LC/Cam_Left_registered.tif",
    description="beads_11mu_0",
)


beads_01mu_0 = NamedDatasetInfo(
    Path(
        "/g/hufnagel/LF/LenseLeNet_Microscope/20191030_Beads_massiveGT/Beads_1micron/2019-10-30_05.20.28_withSimultIllum"
    ),
    "stack_0_channel_0/TP_*/RC_rectified/Cam_Right_1_rectified.tif",
    "stack_1_channel_1/TP_*/LC/Cam_Left_registered.tif",
    description="beads_01mu_0",
)

beads_01mu_1 = NamedDatasetInfo(
    Path(
        "/g/hufnagel/LF/LenseLeNet_Microscope/20191030_Beads_massiveGT/Beads_1micron/2019-10-30_05.54.12_withSimultIllum"
    ),
    "stack_0_channel_0/TP_*/RC_rectified/Cam_Right_1_rectified.tif",
    "stack_1_channel_1/TP_*/LC/Cam_Left_registered.tif",
    description="beads_01mu_1",
)

beads_01mu_2 = NamedDatasetInfo(
    Path(GHUFNAGELLFLenseLeNet_Microscope) / "/20191030_Beads_massiveGT/Beads_1micron/2019-10-30_06.25.24",
    "stack_0_channel_0/TP_*/RC_rectified/Cam_Right_1_rectified.tif",
    "stack_1_channel_1/TP_*/LC/Cam_Left_registered.tif",
    description="beads_1mu_2",
)

beads_01mu_3 = NamedDatasetInfo(
    Path(GHUFNAGELLFLenseLeNet_Microscope) / "/20191030_Beads_massiveGT/Beads_1micron/2019-10-30_06.44.56",
    "stack_0_channel_0/TP_*/RC_rectified/Cam_Right_1_rectified.tif",
    "stack_1_channel_1/TP_*/LC/Cam_Left_registered.tif",
    description="beads_1mu_3",
)

beads_01mu_4 = NamedDatasetInfo(
    Path(GHUFNAGELLFLenseLeNet_Microscope) / "/20191030_Beads_massiveGT/Beads_1micron/2019-10-30_07.04.52",
    "stack_0_channel_0/TP_*/RC_rectified/Cam_Right_1_rectified.tif",
    "stack_1_channel_1/TP_*/LC/Cam_Left_registered.tif",
    description="beads_1mu_4",
)

beads_01mu_5 = NamedDatasetInfo(
    Path(GHUFNAGELLFLenseLeNet_Microscope) / "/20191030_Beads_massiveGT/Beads_1micron/2019-10-30_07.23.55",
    "stack_0_channel_0/TP_*/RC_rectified/Cam_Right_1_rectified.tif",
    "stack_1_channel_1/TP_*/LC/Cam_Left_registered.tif",
    description="beads_1mu_5",
)

beads_4mu_0 = NamedDatasetInfo(
    Path(GHUFNAGELLFLenseLeNet_Microscope) / "/20191030_Beads_massiveGT/Beads_4micron/2019-10-30_08.15.32",
    "stack_0_channel_0/TP_*/RC_rectified/Cam_Right_1_rectified.tif",
    "stack_1_channel_1/TP_*/LC/Cam_Left_registered.tif",
    description="beads_4mu_0",
)

beads_4mu_1 = NamedDatasetInfo(
    Path(GHUFNAGELLFLenseLeNet_Microscope) / "/20191030_Beads_massiveGT/Beads_4micron/2019-10-30_08.34.35",
    "stack_0_channel_0/TP_*/RC_rectified/Cam_Right_1_rectified.tif",
    "stack_1_channel_1/TP_*/LC/Cam_Left_registered.tif",
    description="beads_4mu_1",
)

beads_4mu_2 = NamedDatasetInfo(
    Path(GHUFNAGELLFLenseLeNet_Microscope) / "/20191030_Beads_massiveGT/Beads_4micron/2019-10-30_08.53.53",
    "stack_0_channel_0/TP_*/RC_rectified/Cam_Right_1_rectified.tif",
    "stack_1_channel_1/TP_*/LC/Cam_Left_registered.tif",
    description="beads_4mu_2",
)


beads_4mu_3 = NamedDatasetInfo(
    Path(GHUFNAGELLFLenseLeNet_Microscope) / "/20191030_Beads_massiveGT/Beads_4micron/2019-10-30_09.14.05",
    "stack_0_channel_0/TP_*/RC_rectified/Cam_Right_1_rectified.tif",
    "stack_1_channel_1/TP_*/LC/Cam_Left_registered.tif",
    description="beads_4mu_3",
)

beads_01mix4_0 = NamedDatasetInfo(
    Path(GHUFNAGELLFLenseLeNet_Microscope) / "/20191031_Beads_MixedSizes/Beads_mixed01and4microns/2019-10-31_02.27.33",
    "stack_0_channel_0/TP_*/RC_rectified/Cam_Right_1_rectified.tif",
    "stack_1_channel_1/TP_*/LC/Cam_Left_registered.tif",
    description="beads_01mix4_0",
)

beads_01highc_0 = NamedDatasetInfo(
    Path(
        "/g/hufnagel/LF/LenseLeNet_Microscope/20191031_Beads_MixedSizes/Beads_01micron_highConcentration/2019-10-31_02.57.02"
    ),
    "stack_0_channel_0/TP_*/RC_rectified/Cam_Right_1_rectified.tif",
    "stack_1_channel_1/TP_*/LC/Cam_Left_registered.tif",
    description="beads_01highc_0",
)

beads_01highc_1 = NamedDatasetInfo(
    Path(
        "/g/hufnagel/LF/LenseLeNet_Microscope/20191031_Beads_MixedSizes/Beads_01micron_highConcentration/2019-10-31_03.01.49"
    ),
    "stack_0_channel_0/TP_*/RC_rectified/Cam_Right_1_rectified.tif",
    "stack_1_channel_1/TP_*/LC/Cam_Left_registered.tif",
    description="beads_01highc_1",
)

beads_01highc_2 = NamedDatasetInfo(
    Path(
        "/g/hufnagel/LF/LenseLeNet_Microscope/20191031_Beads_MixedSizes/Beads_01micron_highConcentration/2019-10-31_04.57.13"
    ),
    "stack_0_channel_0/TP_*/RC_rectified/Cam_Right_1_rectified.tif",
    "stack_1_channel_1/TP_*/LC/Cam_Left_registered.tif",
    description="beads_01highc_2",
)
