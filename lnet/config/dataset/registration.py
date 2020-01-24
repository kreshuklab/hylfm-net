from pathlib import Path

import torch.nn.functional
from scipy.ndimage import affine_transform

# adapted from https://github.com/constantinpape/elf/blob/7b7cd21e632a07876a1302dad92f8d7c1929b37a/elf/transformation/affine.py#L162
import numpy


# commented notes on data organisation from "/g/kreshuk//LF_partially_restored/LenseLeNet_Microscope/README_InfoOnDataStructure.txt":

# Information on the data:
# file structure is as follows. There are different crop types, depending on how fast we imaged (when imaging fast the image was already cropped during the imaging and therefore not during copying), and where the heart was positioned in the image (when we slided the static heart with the stage we needed a bigger cropping region). In general there is a trade off when choosing the size of the cropping region: the smaller the better the rectification, but complete heart has to remain in the FOV.
# //--------------------------GENERAL INFO ON CROPPING AND AFFINE TRAFOS------------------------------//
# Cropping parameters were: (for light field RL reconstructions)
# 1) Heart_tightCrop		[250,300,1550,1350] %[xmin,ymin, width, height]
# 2) staticHeartFOV		[100,400,1850,1250]
# 3) wholeFOV				[450,450,1150,1150] %(sorry for naming, relict of former exps)
# 4) fast_cropped_8ms		[0,0,1350,1350]
# 5) fast_cropped_6ms		[0,0,1024,1024]

# //--------------------------------------------------------//
# For all these cropping sizes there are bead data with the same corresponding sizes, which are used to find the correct registration/ affine transformation. This affine transformation is stored in an xml file. The paths of the respective files are the following:
from typing import List, Tuple, Optional, Sequence, Union

from lnet.utils.affine import (
    scipy_form2torch_form_2d,
    scipy_form2torch_form_3d,
    inv_scipy_form2torch_form_2d,
    inv_scipy_form2torch_form_3d,
)

Heart_tightCrop_xml_path = Path(
    "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/beads/after_fish2/definitelyNotMoving/Heart_tightCrop/200msExp/2019-12-09_22.23.27/dataset_Heart_tightCrop.xml"
)
staticHeartFOV_xml_path = Path(
    "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/beads/after_fish2/definitelyNotMoving/staticHeartFOV/200msExp/2019-12-09_22.23.27/dataset_staticHeartFOV.xml"
)
wholeFOV_xml_path = Path(
    "/g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/beads/after_fish2/definitelyNotMoving/staticHeartFOV/200msExp/2019-12-09_22.23.27/dataset_staticHeartFOV.xml"
)
fast_cropped_6ms_xml_path = Path(
    "/g/kreshuk//LF_partially_restored/LenseLeNet_Microscope/20191203_dynamic_staticHeart_tuesday/beads_afterStaticHeart/fast_cropped_6ms/2019-12-03_10.47.58/dataset_fast_cropped_6ms.xml"
)
fast_cropped_8ms_xml_path = Path(
    "/g/kreshuk//LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/beads/after_fish2/definitelyNotMoving/fast_cropped_8ms/200msExp/2019-12-09_22.13.48/dataset_fast_cropped_8ms.xml"
)

# //--------------------------INFO FOR REGISTRATION------------------------------//
#
# To run the registrations on the cluster, one has to use the following files:
# 1) Heart_tightCrop:
# files:
# /g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/files_to_register_Heart_tightCrop.txt
#
# affine transformation xml:
# /g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/beads/after_fish2/definitelyNotMoving/Heart_tightCrop/200msExp/2019-12-09_22.23.27/dataset_Heart_tightCrop.xml
#
# input params for output size:
# new long[]{0, 0, 0},
# new long[]{1462, 1272, 837},
# new long[]{1, 1, 1},
#
# 2) staticHeartFOV:
# files:
# /g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/files_to_register_staticHeartFOV.txt
#
# affine transformation xml:
# /g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/beads/after_fish2/definitelyNotMoving/staticHeartFOV/200msExp/2019-12-09_22.23.27/dataset_staticHeartFOV.xml
#
# input params for output size:
# new long[]{0, 0, 0},
# new long[]{1766, 1177, 837},
# new long[]{1, 1, 1},
#
# 3) fast_cropped_8ms
# affine transformation xml:
# /g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/beads/after_fish2/definitelyNotMoving/fast_cropped_8ms/200msExp/2019-12-09_22.13.48/dataset_fast_cropped_8ms.xml
#
# //--------------------------INFO FOR NETWORK INPUT------------------------------//
#
# For the data of the static heart. The respective files for registered LS stacks and rectified LF image can be found in the following folders:
# 1) registered LS stacks: path always contains 'channel_0' & 'RC_rectified\Cam_Right_001_rectified.tif'
# 2) rectified LF image: path always contains 'channel_1' & 'LC\Cam_Left_registered.tif'
#
# Current experiments of the static heart are the following (all folders might contain cropping regions of 'Heart_tightCrop', and/or 'staticHeartFOV').
# Completely ready (rectified and registered): nothing yet
#
# Almost ready (already copied and rectified, currently registering):
# 1) /g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish2/static
# 2) /g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish1/static/Heart_tightCrop
#
# In process of copying/rectification:
# 3) /g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191208_dynamic_static_heart/fish3/static
# 4) /g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191207_StaticHeart/fish1/static
# 5) /g/kreshuk/LF_partially_restored/LenseLeNet_Microscope/20191203_dynamic_staticHeart_tuesday/fish1/static
#
# //--------------------------INFOS ON FISH DATA------------------------------------//
# 1) Probably best data, both chambers visible, golden experiment worked: 20191208_dynamic_static_heart\fish2
# 2) Only one chamber visible, but this chamber at least crip: 			20191208_dynamic_static_heart\fish1
# 3) Both chambers visible, but more scattering than fish1:				20191208_dynamic_static_heart\fish3
# 4) Not sure if all files are completely static,
# 	fish needed to be silenced several times
# 	-> have to check that heart is really static: 						20191207_StaticHeart\fish1
# 																		20191203_dynamic_staticHeart_tuesday\fish1
#
#
#
# //--------------------------INFOS ON SINGLE SLICE VALIDATION------------------------------------//
# Unfortunately the naming here is going to be a little weird. This has to do with the configuration of the ETL, how RL is reconstructing (orientation in z) and the fact, that in the affine transformations, a flip in z had to be precomputed. Therefore during the saving from h5 to tiff, the stack is flipped in z... For a dynamic sample and single planes this is not possible, because the temporal information would be lost (mixed)..
# Therefore the flip is done when creating the stack with zeros everywhere, except in the plane of interest. This leads to a weird naming (didnt have this in mind during the experiment). So, files with 'plane_100' actually correspond to stacks with single plane validation in plane (241-100+1 = 142).
#


class BDVTransform(torch.nn.Module):
    mode_from_order = {0: "nearest", 2: "bilinear"}

    def __init__(
        self,
        affine_transforms: List[List[float]],
        xml_path: Path,
        output_shape: Optional[Union[Tuple[int, int], Tuple[int, int, int]]] = None,
        order: int = 0,
        additional_transforms_left: Sequence[numpy.ndarray] = tuple(),
        additional_transforms_right: Sequence[numpy.ndarray] = tuple(),
    ):
        super().__init__()
        assert output_shape is None or isinstance(output_shape, tuple)
        self.mode = self.mode_from_order.get(order, None)

        # xml path for reference only
        self.affine_transforms = affine_transforms
        self.trf_matrix = self.concat_affine_matrices(
            list(additional_transforms_left)
            + [self.bdv_trafo_to_affine_matrix(at) for at in affine_transforms]
            + list(additional_transforms_right)
        )
        self.inv_trf_matrix = numpy.linalg.inv(self.trf_matrix)
        self.output_shape = output_shape
        self.order = order
        self.affine_grid_size = None

    @staticmethod
    def bdv_trafo_to_affine_matrix(trafo):
        """ Translate bdv transformation (XYZ) to affine matrix (ZYX)
        """
        assert len(trafo) == 12

        sub_matrix = numpy.zeros((3, 3))
        sub_matrix[0, 0] = trafo[10]
        sub_matrix[0, 1] = trafo[9]
        sub_matrix[0, 2] = trafo[8]

        sub_matrix[1, 0] = trafo[6]
        sub_matrix[1, 1] = trafo[5]
        sub_matrix[1, 2] = trafo[4]

        sub_matrix[2, 0] = trafo[2]
        sub_matrix[2, 1] = trafo[1]
        sub_matrix[2, 2] = trafo[0]

        shift = [trafo[11], trafo[7], trafo[3]]

        matrix = numpy.zeros((4, 4))
        matrix[:3, :3] = sub_matrix
        matrix[:3, 3] = shift
        matrix[3, 3] = 1

        return matrix

    @staticmethod
    def concat_affine_matrices(matrices: List[numpy.ndarray]):
        assert all(m.shape == (4, 4) for m in matrices), [m.shape for m in matrices]
        ret = matrices[0]
        for m in matrices[1:]:
            ret = ret.dot(m)

        return ret

    def _forward(
        self,
        ipt: Union[torch.Tensor, numpy.ndarray],
        matrix: numpy.ndarray,
        inv_matrix: numpy.ndarray,
        output_shape: Optional[Tuple[int, int, int]] = None,
        order: Optional[int] = None,
    ):
        output_shape = output_shape or self.output_shape
        order = order or self.order
        mode = self.mode_from_order[order]
        if isinstance(ipt, numpy.ndarray):
            assert len(ipt.shape) == 3, ipt.shape

            return affine_transform(ipt, matrix, output_shape=output_shape, order=order or self.order)
        elif isinstance(ipt, torch.Tensor):
            assert ipt.shape[0] == 1
            assert ipt.shape[1] == 1
            if len(ipt.shape) == 4:
                torch_form = inv_scipy_form2torch_form_2d(inv_matrix, ipt.shape[2:], output_shape)
            elif len(ipt.shape) == 5:
                torch_form = inv_scipy_form2torch_form_3d(inv_matrix, ipt.shape[2:], output_shape)
            else:
                raise ValueError(ipt.shape)

            affine_grid_size = tuple(ipt.shape[:2]) + output_shape
            if self.affine_grid_size != affine_grid_size:
                self.affine_torch_grid = torch.nn.functional.affine_grid(
                    theta=torch_form, size=affine_grid_size, align_corners=False
                )

            on_cuda = False and ipt.is_cuda
            if not on_cuda and ipt.is_cuda:
                ipt = ipt.to(torch.device("cpu"))

            self.affine_torch_grid = self.affine_torch_grid.to(ipt)

            return torch.nn.functional.grid_sample(ipt, self.affine_torch_grid, align_corners=False, mode=mode)
        else:
            raise TypeError(type(ipt))

    def forward(self, ipt: Union[torch.Tensor, numpy.ndarray], **kwargs):
        return self._forward(ipt, matrix=self.inv_trf_matrix, inv_matrix=self.trf_matrix, **kwargs)

    def forward_with_inverse(self, ipt: Union[torch.Tensor, numpy.ndarray], **kwargs):
        return self._forward(ipt, matrix=self.trf_matrix, inv_matrix=self.inv_trf_matrix, **kwargs)



class Heart_tightCrop_Transform(BDVTransform):
    def __init__(self, **kwargs):
        super().__init__(
            affine_transforms=[
                [
                    0.9991582789246924,
                    -5.0044080331464046e-5,
                    -0.0029455039171397412,
                    1.6498217977948832,
                    -1.9984844050658667e-4,
                    0.9981854168698345,
                    -0.004012925185499846,
                    2.814817686952966,
                    0.0018440989086057481,
                    0.00187876502647589,
                    1.0061875624214813,
                    -5.260792978642528,
                ],
                [
                    1.0010603179057838,
                    2.540114980423077e-4,
                    0.0032125753688346053,
                    -1.9533596838155072,
                    4.870729161447723e-4,
                    1.0016229528348368,
                    0.00430000035201473,
                    -2.934051095325841,
                    -0.0015001383764088087,
                    -0.0019428063338319194,
                    0.9924813629093507,
                    5.471897525039784,
                ],
                [
                    0.9998319189065197,
                    -2.157894534625873e-5,
                    2.1808625761139614e-5,
                    0.09389906857605193,
                    -3.2207625716938455e-4,
                    0.9999752975405105,
                    -4.891449880310803e-4,
                    0.2534155246709321,
                    -9.385312006091003e-6,
                    1.4986500573121512e-4,
                    1.0009111243042954,
                    -0.4208044344377996,
                ],
                [
                    0.9989514005242527,
                    -1.0060954564705695e-4,
                    -0.003382386049290304,
                    1.885774227945873,
                    -3.0805701703008566e-4,
                    0.9982935199106294,
                    -0.0033130477571636976,
                    2.7015412156097165,
                    0.0019296050823401046,
                    0.0014529888710234618,
                    1.0054127594000486,
                    -4.6136712029424585,
                ],
                [
                    1.0001845590539113,
                    -1.820265498351886e-4,
                    -6.600012030655761e-5,
                    0.06796864143361693,
                    1.0858367255225217e-4,
                    1.0001293389628223,
                    -5.417138945726435e-4,
                    0.007332388567066583,
                    -3.7990305655475097e-4,
                    3.9375825610006885e-4,
                    1.001516540361107,
                    -0.622022039259079,
                ],
                [
                    1.001362620955955,
                    -2.2476134566537444e-4,
                    0.001434006491835542,
                    -1.0295317749232165,
                    3.013035013636545e-4,
                    1.0018467283097185,
                    0.004385146373026874,
                    -3.0872715032029197,
                    -0.0012135446739387412,
                    -0.0018358676879072756,
                    0.9950556568838482,
                    4.060025750266694,
                ],
                [
                    1.0024697437231223,
                    -0.0023985273865968246,
                    -0.02592079668834753,
                    3.252997945897662,
                    -0.0014919430102590383,
                    1.0022222829152838,
                    0.0058524955556375335,
                    43.89095474391481,
                    0.008191835682817445,
                    -0.0014897548899628515,
                    1.0126246633717553,
                    -6.051020712374503,
                ],
                [
                    0.97669,
                    0.0076755,
                    0.0042258,
                    -95.112,
                    -0.0061276,
                    0.97912,
                    0.03892,
                    -134.1098,
                    0.007308,
                    0.0073582,
                    1.1682,
                    -92.7323,
                ],
                [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 3.4185, 0.0],
            ],
            xml_path=Heart_tightCrop_xml_path,
            **kwargs,
        )


class staticHeartFOV_Transform(BDVTransform):
    def __init__(self, **kwargs):
        super().__init__(
            affine_transforms=[
                [
                    1.000045172184472,
                    -6.440948265626484e-4,
                    -0.0037246544505502403,
                    1.6647525184522693,
                    -3.741111751453333e-4,
                    0.9997241695263583,
                    -7.727988497216694e-6,
                    0.5482936082360137,
                    6.417439009031318e-4,
                    7.834754261221826e-5,
                    1.0024816523664135,
                    -2.0884853522301463,
                ],
                [
                    1.0031348487012806,
                    -2.4393612341215746e-4,
                    -0.022354095904371995,
                    5.848116160919745,
                    -5.688306131898453e-4,
                    1.0035215202352126,
                    0.005454826549562322,
                    -2.643832484309726,
                    0.009525454800378438,
                    -0.0040831532456764375,
                    1.0083740999442286,
                    -4.757593435405894,
                ],
                [
                    0.97669,
                    0.0076755,
                    0.0042258,
                    -95.112,
                    -0.0061276,
                    0.97912,
                    0.03892,
                    -134.1098,
                    0.007308,
                    0.0073582,
                    1.1682,
                    -92.7323,
                ],
                [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 3.4185, 0.0],
            ],
            xml_path=staticHeartFOV_xml_path,
            **kwargs,
        )


class wholeFOV_Transform(BDVTransform):
    def __init__(self, **kwargs):
        super().__init__(
            affine_transforms=[
                [
                    0.9998417655219239,
                    -1.0649149704165834e-4,
                    -6.867701462479344e-4,
                    0.37590999404288056,
                    -5.002613071260388e-4,
                    1.0000032813545794,
                    8.061629477923528e-5,
                    0.21173749665457114,
                    0.0013881473337330314,
                    1.4598056369385004e-4,
                    0.9984732937104063,
                    -0.23659368825458338,
                ],
                [
                    0.9995638566248812,
                    7.682263504959659e-4,
                    0.0020138351259405022,
                    -0.7886759188097785,
                    0.002918260985264695,
                    1.0017903042862641,
                    -0.002294791022167828,
                    -2.156537036456151,
                    -0.022226519870376035,
                    -0.009315650755218403,
                    1.009644675020266,
                    15.142082260520157,
                ],
                [
                    1.0001500404942196,
                    2.1472856285411563e-4,
                    1.8874243033192004e-4,
                    -0.280571930436458,
                    -6.992448701078586e-5,
                    1.0002552353653096,
                    4.200978328661113e-4,
                    -0.2798317155453338,
                    0.0011755238504975317,
                    -6.110109821577869e-4,
                    0.9977788261920486,
                    0.6182188524479685,
                ],
                [
                    1.0025175741367622,
                    6.174365768619616e-4,
                    -0.0017792662110667574,
                    -1.367250793189235,
                    0.0023839289550945386,
                    1.0013838979893162,
                    3.541361388749152e-4,
                    -2.297239941807359,
                    0.0031901857147841797,
                    -8.095444369189502e-4,
                    0.9982510914673763,
                    -0.7988882839412716,
                ],
                [
                    0.9973170930308844,
                    -8.356232274227848e-4,
                    0.0015445869043996564,
                    1.6731476169665824,
                    -0.0023156559139415702,
                    0.998338153844677,
                    -7.557240197089434e-4,
                    2.5835373016043777,
                    -0.004452846773506765,
                    0.0014411842070933343,
                    1.0042132041845395,
                    0.13089273474828747,
                ],
                [
                    1.0000071744446184,
                    -4.82849496709715e-6,
                    9.765732650221969e-5,
                    -0.03496926928437178,
                    -5.228618584191114e-5,
                    1.000018654958159,
                    1.2995048492377563e-4,
                    -0.027209187582010165,
                    7.22914999300049e-5,
                    2.7215339092746645e-5,
                    0.9998954823124625,
                    -0.018249062421577807,
                ],
                [
                    1.0000725577838656,
                    -1.29767841362559e-4,
                    8.840954118647092e-4,
                    -0.31601302665062975,
                    -4.4016440953112195e-4,
                    1.0001870462447495,
                    0.0012635467404133968,
                    -0.24191803171217455,
                    8.134618337626213e-4,
                    3.0615903190084126e-4,
                    0.9989568916038375,
                    -0.16384972574530204,
                ],
                [
                    1.0004982495624928,
                    -5.347748924730422e-4,
                    -0.002241468562732871,
                    0.75641779796241,
                    -0.001867167711927428,
                    0.9980569861154626,
                    7.752669266069475e-4,
                    2.1641121189209316,
                    0.0198747808280692,
                    0.008716320243568774,
                    0.9928196863957867,
                    -14.529850993181118,
                ],
                [
                    1.0000851293251374,
                    -4.167960675736972e-5,
                    -7.811302332578192e-4,
                    0.2386479047513763,
                    -2.8177059676961905e-4,
                    1.0001505852122112,
                    -4.8903086335231217e-5,
                    0.05245873241772043,
                    0.0017201577194763504,
                    2.0968531980055196e-4,
                    0.99811689515336,
                    -0.33263194424705583,
                ],
                [
                    1.0019956042854006,
                    0.0015330238853147203,
                    5.10242017551151e-4,
                    -2.2997528226605928,
                    0.0050132811519601455,
                    1.0032888728896525,
                    -0.0013810723598523503,
                    -4.582679428870678,
                    -0.018245048908209948,
                    -0.01085403241603045,
                    1.0060252751976488,
                    15.127317971968829,
                ],
                [
                    0.9975549351085755,
                    -6.900315538870438e-4,
                    0.001623785480176771,
                    1.4363613470066294,
                    -0.002174658561239581,
                    0.9984891259742981,
                    -8.129618761158002e-4,
                    2.443327604562551,
                    -0.0042923329636386206,
                    0.0014372985400422133,
                    1.0038543706088774,
                    0.1707551823158092,
                ],
                [
                    1.0024003329546827,
                    6.839184624593545e-4,
                    -0.0016264099569507826,
                    -1.406212162217873,
                    0.0021511350338339974,
                    1.0015374398962855,
                    9.525180234722576e-4,
                    -2.4981809728195263,
                    0.0041846314491129,
                    -0.0013980227058144885,
                    0.9963347312022911,
                    -0.2034310602369239,
                ],
                [
                    0.997150223279827,
                    -9.302616779973354e-4,
                    0.0012933978056324418,
                    1.8565375902253212,
                    -0.002358058922914878,
                    0.9987067940004971,
                    6.336497537877597e-4,
                    1.9677587547725377,
                    -0.004906214018464217,
                    0.001602316169046321,
                    1.0055207614415265,
                    -0.21139653230094949,
                ],
                [
                    0.9990087724216514,
                    3.0290313015609465e-4,
                    -0.01538269131085553,
                    18.132461015496517,
                    -4.368204399164435e-4,
                    1.00156390437887,
                    0.004193482059189924,
                    -16.42955022422457,
                    0.010880402921432629,
                    0.01225699056146277,
                    1.0143851032128068,
                    -30.49093336067544,
                ],
                [
                    0.9998763934501763,
                    -0.0010891310450819906,
                    0.00253370847820035,
                    -0.6115113860549377,
                    7.675130470486004e-4,
                    0.9983682901916651,
                    -7.40670235632776e-4,
                    0.8275234429996294,
                    -0.00947710327402089,
                    0.0038009121698615866,
                    0.9993213751411238,
                    3.1765203213020143,
                ],
                [
                    1.0001499960607572,
                    0.0014027361439267177,
                    -0.0017760750843483504,
                    -0.10921947468546678,
                    -1.4673922853670316e-4,
                    0.9992407191400258,
                    -0.0017297884331933565,
                    0.9394132446795214,
                    0.024767938375552963,
                    -0.00398032235341257,
                    0.9783021841535382,
                    -7.444609223601558,
                ],
                [
                    1.0003559629383398,
                    2.9327823350429983e-4,
                    6.428006590138689e-4,
                    -0.5171440459931798,
                    -4.050787570647998e-4,
                    0.9996835486510609,
                    -2.6101507441633878e-5,
                    0.3307699115452033,
                    -0.007875800046626481,
                    -0.002452377176303182,
                    0.9930269401854842,
                    7.089568501996945,
                ],
                [
                    1.0008586374814403,
                    0.0017016621421833808,
                    0.0011200077040748062,
                    -0.2440484013785469,
                    4.817641577129386e-4,
                    0.9983909483762317,
                    0.002322117596739517,
                    -1.191038991556635,
                    -0.012612826427677736,
                    0.0017988091415114254,
                    1.0142720444336737,
                    6.527121708955718,
                ],
                [
                    0.97958,
                    0.0047483,
                    -0.01109,
                    -151.2572,
                    -0.0074967,
                    0.98373,
                    0.049058,
                    -134.7033,
                    0.013631,
                    -0.0030357,
                    1.1662,
                    -75.4176,
                ],
                [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 3.4185, 0.0],
            ],
            xml_path=wholeFOV_xml_path,
            **kwargs,
        )


class fast_cropped_6ms_Transform(BDVTransform):
    def __init__(self, **kwargs):
        super().__init__(
            affine_transforms=[
                [
                    1.0002459794124785,
                    -2.2800776616194434e-4,
                    7.06635354005253e-4,
                    -0.310507600323082,
                    -1.6436277044339917e-4,
                    1.000130416649231,
                    1.2675415989102842e-4,
                    0.02071198230079377,
                    -0.006819804116110938,
                    0.002760002365515835,
                    1.005591969344203,
                    1.3516919745339329,
                ],
                [
                    0.9997554273413035,
                    2.3605194775509156e-4,
                    -6.924948788607131e-4,
                    0.30161248423846176,
                    1.7307578986511582e-4,
                    0.9998857576307465,
                    -1.2431688124025126e-4,
                    -0.033720084165848305,
                    0.006789030910848358,
                    -0.002745119133451576,
                    0.9944079280754539,
                    -1.336539711845158,
                ],
                [
                    1.0002999931827274,
                    -1.7424955178460828e-4,
                    7.747523582806886e-4,
                    -0.3886584055398829,
                    -7.526945519961514e-5,
                    1.0002724045539166,
                    1.2600866536445355e-4,
                    -0.09406334524547129,
                    -0.0067096819112248745,
                    0.0027545563116734037,
                    1.0053461443220024,
                    1.3787048653452845,
                ],
                [
                    1.0003045576351692,
                    0.0013508434831580461,
                    -1.6493465412205095e-4,
                    -0.7432324164894379,
                    4.845596049328561e-4,
                    1.0013082528497923,
                    9.678886678765577e-4,
                    -1.408435427994669,
                    0.008001777014911319,
                    -0.0038694726723087925,
                    0.9920015459459663,
                    -0.543309646302159,
                ],
                [
                    0.9995007975492034,
                    -0.001160436056024711,
                    -4.656763320559507e-4,
                    1.0034378137190076,
                    -3.070602992248084e-4,
                    0.9984231547086668,
                    -9.998896739684166e-4,
                    1.4002372648270065,
                    -0.0028081724905490166,
                    9.91407312774693e-4,
                    1.0020357549233985,
                    0.40823789022980733,
                ],
                [
                    1.0001673226422105,
                    -2.5007202083074224e-4,
                    2.3942102071763976e-4,
                    -0.08197644365428648,
                    -3.0761646270283387e-4,
                    1.000149108002461,
                    2.6453062735372666e-4,
                    0.05459382373050764,
                    -0.005646233706300736,
                    0.002997753393853305,
                    1.006828811526157,
                    0.032925484769386906,
                ],
                [
                    1.0000641426395687,
                    3.991412305563949e-5,
                    3.8494220770142213e-4,
                    -0.18681276543478734,
                    3.7993739239118724e-5,
                    1.0002093325561061,
                    -1.7142164015812985e-5,
                    -0.11247435589269912,
                    3.9713356658404464e-4,
                    -4.876427889456174e-5,
                    0.9992308054006369,
                    0.07106374786050755,
                ],
                [
                    1.0006635981556136,
                    3.2484007182119126e-4,
                    -0.0013487090395483512,
                    -0.19023673737013777,
                    4.5599607110634517e-4,
                    1.0020959354993126,
                    0.002001401251431046,
                    -2.0316561939875983,
                    0.009072407722233547,
                    -0.002635303624325553,
                    0.9923448975374671,
                    -1.5472322100460143,
                ],
                [
                    0.9993290054471409,
                    2.2427309163849024e-4,
                    6.354544506468878e-4,
                    0.0707471753866127,
                    -1.0642995073706009e-4,
                    0.999778393862275,
                    9.938278259580958e-4,
                    -0.14354387390236484,
                    -0.003938578709303977,
                    0.003172926018743711,
                    1.0012307830969316,
                    0.09969919601110715,
                ],
                [
                    0.9987285932863739,
                    1.5557144455591472e-4,
                    -0.013691022603613988,
                    120.6883922598818,
                    -4.526758033175239e-4,
                    1.000082580133885,
                    0.0019164519758949493,
                    88.0206519687398,
                    0.007785551759521633,
                    0.006814207789026994,
                    1.0095319730590382,
                    -20.111755129095247,
                ],
                [
                    0.9998763934501763,
                    -0.0010891310450819906,
                    0.00253370847820035,
                    -0.6115113860549377,
                    7.675130470486004e-4,
                    0.9983682901916651,
                    -7.40670235632776e-4,
                    0.8275234429996294,
                    -0.00947710327402089,
                    0.0038009121698615866,
                    0.9993213751411238,
                    3.1765203213020143,
                ],
                [
                    1.0001499960607572,
                    0.0014027361439267177,
                    -0.0017760750843483504,
                    -0.10921947468546678,
                    -1.4673922853670316e-4,
                    0.9992407191400258,
                    -0.0017297884331933565,
                    0.9394132446795214,
                    0.024767938375552963,
                    -0.00398032235341257,
                    0.9783021841535382,
                    -7.444609223601558,
                ],
                [
                    1.0003559629383398,
                    2.9327823350429983e-4,
                    6.428006590138689e-4,
                    -0.5171440459931798,
                    -4.050787570647998e-4,
                    0.9996835486510609,
                    -2.6101507441633878e-5,
                    0.3307699115452033,
                    -0.007875800046626481,
                    -0.002452377176303182,
                    0.9930269401854842,
                    7.089568501996945,
                ],
                [
                    1.0008586374814403,
                    0.0017016621421833808,
                    0.0011200077040748062,
                    -0.2440484013785469,
                    4.817641577129386e-4,
                    0.9983909483762317,
                    0.002322117596739517,
                    -1.191038991556635,
                    -0.012612826427677736,
                    0.0017988091415114254,
                    1.0142720444336737,
                    6.527121708955718,
                ],
                [
                    0.97958,
                    0.0047483,
                    -0.01109,
                    -151.2572,
                    -0.0074967,
                    0.98373,
                    0.049058,
                    -134.7033,
                    0.013631,
                    -0.0030357,
                    1.1662,
                    -75.4176,
                ],
                [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 3.4185, 0.0],
            ],
            xml_path=fast_cropped_6ms_xml_path,
            **kwargs,
        )


class fast_cropped_8ms_Transform(BDVTransform):
    def __init__(self, **kwargs):
        super().__init__(
            affine_transforms=[
                [
                    0.9986010487421885,
                    -0.002120810141566837,
                    -0.008149618765723362,
                    4.753590473932092,
                    -7.497126762641354e-4,
                    0.999739497335616,
                    -0.002100267232562155,
                    1.5264448905391126,
                    0.012767265132301005,
                    0.008313927705739728,
                    1.0251251343976073,
                    -23.383246688659526,
                ],
                [
                    1.0014699495221129,
                    0.0022217885570053057,
                    0.007815338643888291,
                    -4.714470966958904,
                    0.0010303617894547048,
                    1.0000489406185478,
                    0.001449929955455748,
                    -1.2415661154926954,
                    -0.012897434787601393,
                    -0.008282360222404195,
                    0.9728093345347644,
                    24.192111553605027,
                ],
                [
                    0.9999756097799303,
                    -3.14068807737589e-5,
                    2.6414434466927684e-6,
                    0.03299776828073804,
                    -7.443462885587118e-6,
                    1.0000205503425315,
                    6.372773046833699e-5,
                    -0.032715113781997054,
                    3.53248611190156e-5,
                    2.2833262680210066e-5,
                    1.000242859147333,
                    -0.1291290449835767,
                ],
                [
                    0.9997562287077025,
                    -1.4690002895491214e-4,
                    -1.3886713774554479e-5,
                    0.23433800490792453,
                    -2.4140517872558393e-4,
                    1.0002056303267557,
                    7.243447875154269e-4,
                    -0.3313980780587243,
                    3.9422039654427526e-4,
                    1.433164031456372e-4,
                    1.0024346602689587,
                    -1.2459354397517026,
                ],
                [
                    0.9977154744301842,
                    -0.002241191411341819,
                    -0.009852533063965996,
                    6.0748338797101615,
                    -2.1704150642624007e-4,
                    1.000024837549752,
                    -1.479204794364181e-4,
                    0.35024621525122906,
                    0.013231836246675045,
                    0.008248997222915135,
                    1.025443845475132,
                    -23.83731141291179,
                ],
                [
                    1.002301101653545,
                    0.001244721528787248,
                    0.006274675516564312,
                    -4.134709347642092,
                    2.737459409621002e-4,
                    1.00055221854111,
                    0.0016732066436690894,
                    -1.2089095422920313,
                    -0.013232668769306219,
                    -0.008141878604109042,
                    0.9701538227526025,
                    25.280673996062564,
                ],
                [
                    1.0036672920493648,
                    -0.001217988458688357,
                    -0.029144491141293156,
                    45.37683487750689,
                    -1.6696390203153297e-4,
                    1.003610058305277,
                    0.011245829556812224,
                    95.22415193683851,
                    0.010100024209689917,
                    -0.005478727449094312,
                    0.9926932232933254,
                    7.776297843932661,
                ],
                [
                    0.97669,
                    0.0076755,
                    0.0042258,
                    -95.112,
                    -0.0061276,
                    0.97912,
                    0.03892,
                    -134.1098,
                    0.007308,
                    0.0073582,
                    1.1682,
                    -92.7323,
                ],
                [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 3.4185, 0.0],
            ],
            xml_path=fast_cropped_8ms_xml_path,
            **kwargs,
        )
