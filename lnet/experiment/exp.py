import pbs3
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
from lnet.utils.stat import DatasetStat
from lnet.experiment.base import ExperimentBase, eps_for_precision, LOSS_NAME, MSSSIM_NAME, BEAD_RECALL, BEAD_PRECISION
from lnet.experiment.config import Config

torch_dtype_to_inferno = {torch.float: "float", torch.float32: "float", torch.half: "half", torch.float16: "half"}


class Experiment(ExperimentBase):
    def __init__(self, config: Union[Path, Config]):
        if isinstance(config, Path):
            self.config_path = config
            config = Config(config)
        else:
            self.config_path = None

        assert isinstance(config, Config)

        self.nnum = 19
        self.Model = config.model.Model
        has_aux = True
        self.additional_model_kwargs = {"final_activation": torch.nn.Sigmoid(), "aux_activation": None}

        self.precision = torch.float
        self.batch_size = 1
        self.eval_batch_size = 3

        self.optimizer_cls = torch.optim.Adam
        self.optimizer_kwargs = {"lr": 1e-4, "eps": eps_for_precision[self.precision]}
        self.max_num_epochs = 5000

        def my_norm(stat: DatasetStat) -> Generator[Transform, None, None]:
            yield Normalize(mean=stat.x_mean, std=stat.x_std, apply_to=[0])
            yield Normalize01Sig(min_=stat.corr_y_min, max_=stat.corr_y_max, apply_to=[1])
            yield Normalize01Sig(min_=stat.corr_y_min, max_=stat.corr_y_max / 2, apply_to=[2])

        def noise(stat: DatasetStat) -> Generator[Transform, None, None]:
            yield AdditiveGaussianNoise(sigma=stat.x_std / 5, apply_to=[0])
            yield AdditiveGaussianNoise(sigma=stat.y_std, apply_to=[1])
            yield AdditiveGaussianNoise(sigma=stat.y_std / 2, apply_to=[2])

        self.train_transforms = [
            noise,
            my_norm,
            RandomRotate(),
            RandomFlipXYnotZ(),
            Lightfield2Channel(nnum=self.nnum),
            Cast(torch_dtype_to_inferno[self.precision]),
        ]
        self.valid_transforms = [
            my_norm,
            Lightfield2Channel(nnum=self.nnum),
            Cast(torch_dtype_to_inferno[self.precision]),
        ]
        self.test_transforms = [
            my_norm,
            Lightfield2Channel(nnum=self.nnum),
            Cast(torch_dtype_to_inferno[self.precision]),
        ]

        self.train_dataset_factory = DatasetFactory(
            DatasetConfig(fish.fish02_LS0_filet, beads.bead00_LS0_35), has_aux=has_aux
        )
        self.train_data_range = range(5, 73)
        self.train_eval_data_range = range(5, 5 + self.eval_batch_size)
        self.valid_dataset_factory = DatasetFactory(DatasetConfig(fish.fish02_LS0_filet), has_aux=has_aux)
        self.valid_data_range = range(4, 5)
        self.test_dataset_factory = DatasetFactory(DatasetConfig(fish.fish02_LS0_filet), has_aux=has_aux)
        self.test_data_range = range(4)

        super().__init__(
            loss_fn=SorensenDiceLoss(channelwise=False, eps=1.0e-4),
            aux_loss_fn=torch.nn.BCEWithLogitsLoss(),
            checkpoint=config.model.checkpoint
            or Path(
                "/g/kreshuk/beuttenm/repos/lensletnet/logs/beads/00_LS0/8/19-09-19_14-28_db5e04e_8-0/models/v0_model_887.pth"
            ),
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
