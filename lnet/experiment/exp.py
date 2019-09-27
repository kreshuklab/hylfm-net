import torch
import torch.nn
import torch.optim
import torch.utils.data

from inferno.extensions.criteria.set_similarity_measures import SorensenDiceLoss
from inferno.io.transform import Transform
from inferno.io.transform.generic import Cast, Normalize
from inferno.io.transform.image import AdditiveGaussianNoise
from pathlib import Path
from typing import Type, Generator, Union

from lnet.dataset_configs import fish, platy, beads
from lnet.utils.datasets import DatasetFactory, DatasetConfig
from lnet.utils.data_transform import (
    Clip,
    Lightfield2Channel,
    RandomFlipXYnotZ,
    RandomRotate,
    EdgeCrop,
    Normalize01Sig,
    Normalize01,
)
from lnet.experiment.base import ExperimentBase, eps_for_precision, LOSS_NAME, MSSSIM_NAME, BEAD_RECALL, BEAD_PRECISION
from lnet.experiment.config import Config

torch_dtype_to_inferno = {torch.float: "float", torch.float32: "float", torch.half: "half", torch.float16: "half"}


class Experiment(ExperimentBase):
    def __init__(self, config: Union[Path, Config]):


        super().__init__(
            config=config,
        )

    @staticmethod
    def score_function(engine):
        score = engine.state.metrics[MSSSIM_NAME]
        # p = experiment.state.metrics.get(BEAD_PRECISION, 0)
        # r = experiment.state.metrics.get(BEAD_RECALL, 0)
        # if p and r:
        #     score += 2 * p * r / (p + r)

        return score

    def run(self):
        super().run()
        if self.config_path is not None:
            with self.config_path.with_suffix(".ran_on.txt").open("a") as f:
                f.write(self.config.log.commit_hash)


def runnable_experiments() -> Generator[Type[Experiment], None, None]:
    yield Experiment
