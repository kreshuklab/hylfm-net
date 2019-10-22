from dataclasses import dataclass
from typing import List

import torch


@dataclass
class Output:
    ipt: torch.Tensor
    tgt: torch.Tensor
    aux_tgt: torch.Tensor
    pred: torch.Tensor
    aux_pred: torch.Tensor
    loss: torch.Tensor
    aux_loss: torch.Tensor
    losses: List[torch.Tensor]
    aux_losses: List[torch.Tensor]
