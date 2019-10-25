LOSS_NAME = "Loss"
AUX_LOSS_NAME = "AuxLoss"
NRMSE_NAME = "NRMSE"
PSNR_NAME = "PSNR"
SSIM_NAME = "SSIM"
MSSSIM_NAME = "MS-SSIM"
BEAD_PRECISION_RECALL = "Bead-Precision-Recall"

from .msssim import MSSSIM, SSIM  # SSIMSkImage != SSIM
from .nrmse import NRMSE
from .psnr import PSNR
