import torch
from inferno.extensions.criteria import SorensenDiceLoss

known_losses = {
    "BCEWithLogitsLoss": [(1.0, torch.nn.BCEWithLogitsLoss())],
    "SorensenDiceLoss": [(1.0, SorensenDiceLoss(channelwise=False, eps=1.0e-4))],
}
