from pathlib import Path

from lnet.config import Config, ModelConfig, LogConfig
from lnet.models import M13dout

val_every = 10
config = Config(
    model=ModelConfig(
        Model=M13dout,
        kwargs={"final_activation": "sigmoid", "aux_activation": None},
        nnum=19,
        precision="float",
        checkpoint=None,
    ),
    log=LogConfig(
        config_path=Path(__file__),
        validate_every_nth_epoch=val_every,
        log_scalars_every=(val_every, "epoch"),
        log_images_every=(val_every, "epoch"),
        log_bead_precision_recall=True,
    ),
    ...
)
