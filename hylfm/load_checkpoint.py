from pathlib import Path

import torch

from hylfm.get_model import get_model
from hylfm.load_old_checkpoint import get_config_for_old_checkpoint


def load_state_from_checkpoint(checkpoint: Path) -> dict:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    state = torch.load(str(checkpoint), map_location=device)
    return state


def load_model_from_state(checkpoint: Path, state: dict):
    config = state.get("config", None)
    if config is None:
        config = get_config_for_old_checkpoint(checkpoint)

    config["checkpoint"] = str(checkpoint)
    model = get_model(**config["model"])
    model.load_state_dict(state["model"], strict=True)
    return model, config


def load_model_from_checkpoint(checkpoint: Path):
    return load_model_from_state(checkpoint, load_state_from_checkpoint(checkpoint))
