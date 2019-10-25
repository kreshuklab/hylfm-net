from dataclasses import dataclass
from typing import List, Optional

import torch


@dataclass
class Output:
    ipt: torch.Tensor
    tgt: torch.Tensor
    pred: torch.Tensor
    loss: torch.Tensor
    losses: List[torch.Tensor]
    voxel_losses: Optional[List[torch.Tensor]] = None


@dataclass
class AuxOutput(Output):
    aux_tgt: Optional[torch.Tensor] = None
    aux_pred: Optional[torch.Tensor] = None
    aux_loss: Optional[torch.Tensor] = None
    aux_losses: Optional[List[torch.Tensor]] = None
