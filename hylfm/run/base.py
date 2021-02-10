import logging
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

import torch
from hylfm.datasets import ConcatDataset, get_collate
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from hylfm import __version__, metrics, settings
from hylfm.checkpoint import Checkpoint, Config, IndependentConfig
from hylfm.datasets.named import get_dataset
from hylfm.hylfm_types import DatasetPart, TransformLike, TransformsPipeline
from hylfm.metrics.base import MetricGroup
from hylfm.model import HyLFM_Net
from hylfm.run.run_logger import RunLogger
from hylfm.sampler import NoCrossBatchSampler
from hylfm.transform_pipelines import get_transforms_pipeline

logger = logging.getLogger(__name__)


class Run:
    name: str
    log_path: Optional[Path]

    def __init__(
        self,
        *,
        config: Config,
        model: Optional[HyLFM_Net],
        dataset_part: DatasetPart,  # todo: one runner, one dataset part!
        independent_config: IndependentConfig,
        name: str,
        # batch_postprocessing: TransformLike,
        # batch_premetric_trf: TransformLike,
        # batch_preprocessing_in_step: TransformLike,
        # dataloader: torch.utils.data.DataLoader,
        # metrics: MetricGroup,
        # model: Optional[torch.nn.Module],
        # name: Optional[str] = None,
        # run_logger: RunLogger,
        # tgt_name: Optional[str],
    ):
        assert isinstance(__version__, str)
        self.dirty = "dirty" in __version__
        if self.dirty:
            warnings.warn(f"uncommited changes in version {__version__}")

        self.config = cfg = config
        self.independent_config = ind_cfg = independent_config
        self.name = name

        if model is not None and torch.cuda.is_available():
            model = model.cuda(0)

        self.model = model
        self.scale = scale = model.get_scale()
        self.shrink = shrink = model.get_shrink()

        self.nnum = nnum = self.model.nnum
        self.z_out = z_out = self.model.z_out
        assert nnum == cfg.model["nnum"]
        assert z_out == cfg.model["z_out"]

        self.dataset_parts = dataset_parts
        self.transforms_pipelines: Dict[DatasetPart, TransformsPipeline] = {
            part: get_transforms_pipeline(
                dataset_name=self.config.dataset,
                dataset_part=part,
                nnum=cfg.model["nnum"],
                z_out=cfg.model["z_out"],
                scale=scale,
                shrink=shrink,
                interpolation_order=self.config.interpolation_order,
            )
            for part in dataset_parts
        }

        self.datasets: Dict[DatasetPart, ConcatDataset] = {
            part: get_dataset(self.config.dataset, part, self.transforms_pipelines[part]) for part in dataset_parts
        }

        self.dataloaders: Dict[DatasetPart, DataLoader] = {
            part: DataLoader(
                dataset=self.datasets[part],
                batch_sampler=NoCrossBatchSampler(
                    self.datasets[part],
                    sampler_class=RandomSampler if part == DatasetPart.train else SequentialSampler,
                    batch_sizes=[self.config.batch_size if part == DatasetPart.train else self.config.eval_batch_size]
                    * len(self.datasets[part].cumulative_sizes),
                    drop_last=False,
                ),
                collate_fn=get_collate(batch_transformation=self.transforms_pipelines[part].batch_preprocessing),
                num_workers=settings.num_workers_train_data_loader,
                pin_memory=settings.pin_memory,
            )
            for part in (DatasetPart.train, DatasetPart.validate)
        }

        self.metric_groups: Dict[DatasetPart, MetricGroup] = self.get_metric_groups()

    def get_metric_groups(self) -> Dict[DatasetPart, MetricGroup]:
        cfg = self.config
        parts = self.dataset_parts

        groups = {}
        if DatasetPart.train in parts:
            groups[DatasetPart.train]: MetricGroup()

        if DatasetPart.validate in parts:
            groups[DatasetPart.validate] = MetricGroup(
                metrics.MSE(),
                metrics.MS_SSIM(
                    channel=1,
                    data_range=cfg.data_range,
                    size_average=True,
                    spatial_dims=self.transforms_pipelines[DatasetPart.validate].spatial_dims,
                    win_size=cfg.win_size,
                    win_sigma=cfg.win_sigma,
                ),
                metrics.NRMSE(),
                metrics.PSNR(data_range=cfg.data_range),
                metrics.SSIM(
                    data_range=cfg.data_range,
                    size_average=True,
                    win_size=cfg.win_size,
                    win_sigma=cfg.win_sigma,
                    channel=1,
                    spatial_dims=self.transforms_pipelines[DatasetPart.validate].spatial_dims,
                ),
                metrics.SmoothL1(),
            )

            if self.transforms_pipelines[DatasetPart.validate].spatial_dims == 3:
                groups[DatasetPart.validate] += MetricGroup(
                    metrics.BeadPrecisionRecall(
                        dist_threshold=3.0,
                        exclude_border=False,
                        max_sigma=6.0,
                        min_sigma=1.0,
                        overlap=0.5,
                        sigma_ratio=3.0,
                        threshold=0.3,  # orig 0.05
                        tgt_threshold=0.3,  # orig 0.05
                        scaling=(2.5, 0.7 * 8 / self.scale, 0.7 * 8 / self.scale),
                    )
                )

        if DatasetPart.test in parts:
            groups[DatasetPart.test] = MetricGroup(
                # on volume
                metrics.BeadPrecisionRecall(
                    dist_threshold=3.0,
                    exclude_border=False,
                    max_sigma=6.0,
                    min_sigma=1.0,
                    overlap=0.5,
                    scaling=(2.5, 0.7 * 8 / self.scale, 0.7 * 8 / self.scale),
                    sigma_ratio=3.0,
                    tgt_threshold=0.3,  # orig: 0.05
                    threshold=0.3,  # orig: 0.05
                ),
                metrics.MSE(),
                metrics.MS_SSIM(
                    channel=1,
                    data_range=cfg.data_range,
                    size_average=True,
                    spatial_dims=3,
                    win_size=cfg.win_size,
                    win_sigma=cfg.win_sigma,
                ),
                metrics.NRMSE(),
                metrics.PSNR(data_range=cfg.data_range),
                metrics.SSIM(
                    data_range=cfg.data_range,
                    size_average=True,
                    win_size=cfg.win_size,
                    win_sigma=cfg.win_sigma,
                    channel=1,
                    spatial_dims=3,
                ),
                metrics.SmoothL1(),
            )
            if not self.independent_config.light_logging:
                groups[DatasetPart.test] += MetricGroup(
                    # along z
                    metrics.MSE(along_dim=1),
                    metrics.MS_SSIM(
                        along_dim=1,
                        channel=1,
                        data_range=cfg.data_range,
                        size_average=True,
                        spatial_dims=2,
                        win_sigma=cfg.win_sigma,
                        win_size=cfg.win_size,
                    ),
                    metrics.NRMSE(along_dim=1),
                    metrics.PSNR(along_dim=1, data_range=cfg.data_range),
                    metrics.SSIM(
                        along_dim=1,
                        channel=1,
                        data_range=cfg.data_range,
                        size_average=True,
                        spatial_dims=2,
                        win_sigma=cfg.win_sigma,
                        win_size=cfg.win_size,
                    ),
                    metrics.SmoothL1(along_dim=1),
                )

        return groups

    def __iter__(self):
        for batch in self._run():
            yield batch

    def run(self):
        for it in self:
            pass

    def _run(self) -> Iterable[Dict[str, Any]]:
        raise NotImplementedError
