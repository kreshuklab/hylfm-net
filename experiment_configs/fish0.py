from pathlib import Path

from lnet.experiment.config import Config, ModelConfig, LogConfig
from lnet.models import M13dout

val_every = 10
config = Config(model=ModelConfig(Model=M13dout, nnum=19, ), log=LogConfig(config_path=Path(__file__), validate_every_nth_epoch=val_every, log_scalars_every=(val_every, "epoch"), log_images_every=(val_every, "epoch"), log_bead_precision_recall=True),)
