import logging
import warnings
from typing import Any, Dict, Iterable, Optional

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from hylfm import __version__, metrics, settings
from hylfm.checkpoint import RunConfig
from hylfm.datasets import ConcatDataset, get_collate
from hylfm.datasets.named import get_dataset
from hylfm.hylfm_types import DatasetPart, TransformsPipeline
from hylfm.metrics.base import MetricGroup
from hylfm.model import HyLFM_Net
from hylfm.run.run_logger import WandbLogger
from hylfm.sampler import NoCrossBatchSampler
from hylfm.transform_pipelines import get_transforms_pipeline

logger = logging.getLogger(__name__)


class Run:
    name: str

    def __init__(
        self,
        *,
        config: RunConfig,
        model: Optional[HyLFM_Net],
        dataset_part: DatasetPart,
        name: str,
        run_logger: WandbLogger,
        scale: Optional[int] = None,
        shrink: Optional[int] = None,
    ):
        assert isinstance(__version__, str)
        self.dirty = "dirty" in __version__
        if self.dirty:
            warnings.warn(f"uncommited changes in version {__version__}")

        self.config = cfg = config
        self.name = name
        self.save_output_to_disk = {
            key: settings.log_dir / name / dataset_part.name / "output_tensors" / key
            for key in (config.save_output_to_disk or [])
        }
        for path in self.save_output_to_disk.values():
            path.mkdir(parents=True, exist_ok=True)

        self.model = model
        if model is None:
            assert scale is not None
            assert shrink is not None
        else:
            assert scale is None
            assert shrink is None
            scale = model.get_scale()
            shrink = model.get_shrink()
            if torch.cuda.is_available():
                model = model.cuda(0)

        self.scale = scale
        self.shrink = shrink

        import wandb

        wandb.summary.update(dict(scale=scale, shrink=shrink))

        self.dataset_part = dataset_part
        self.transforms_pipeline: Optional[TransformsPipeline] = None if model is None else get_transforms_pipeline(
            dataset_name=cfg.dataset,
            dataset_part=dataset_part,
            nnum=self.model.nnum,
            z_out=self.model.z_out,
            scale=scale,
            shrink=shrink,
            interpolation_order=cfg.interpolation_order,
        )

        self.dataset: ConcatDataset = get_dataset(cfg.dataset, dataset_part, self.transforms_pipeline)

        self.dataloader: DataLoader = DataLoader(
            dataset=self.dataset,
            batch_sampler=NoCrossBatchSampler(
                self.dataset,
                sampler_class=RandomSampler if self.dataset_part == DatasetPart.train else SequentialSampler,
                batch_sizes=[cfg.batch_size if self.dataset_part == DatasetPart.train else cfg.batch_size]
                * len(self.dataset.cumulative_sizes),
                drop_last=self.dataset_part == DatasetPart.train,
            ),
            collate_fn=get_collate(batch_transformation=self.transforms_pipeline.batch_preprocessing),
            num_workers=settings.num_workers_data_loader[dataset_part.name],
            pin_memory=settings.pin_memory,
        )
        self.epoch_len = len(self.dataloader)
        assert self.epoch_len
        assert self.epoch_len < 100000 or not self.save_output_to_disk

        self.metric_group: MetricGroup = self.get_metric_group()
        self.run_logger = run_logger

    def get_metric_group(self) -> MetricGroup:
        cfg = self.config

        if self.dataset_part == DatasetPart.train:
            return MetricGroup()

        elif self.dataset_part in (DatasetPart.validate, DatasetPart.test):
            group = [
                # basic metrics for 2 or 3d
                metrics.MSE(),
                metrics.MS_SSIM(
                    channel=1,
                    data_range=cfg.data_range,
                    size_average=True,
                    spatial_dims=self.transforms_pipeline.spatial_dims,
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
                    spatial_dims=self.transforms_pipeline.spatial_dims,
                ),
                metrics.SmoothL1(),
            ]

            if self.transforms_pipeline.spatial_dims == 3:
                group.append(
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
                if self.dataset_part == DatasetPart.test:
                    # along z if basic metrics are 3d
                    group += [
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
                    ]
            return MetricGroup(*group)
        else:
            raise NotImplementedError(self.dataset_part)

    def __iter__(self):
        for batch in self._run():
            yield batch

    def run(self):
        for it in self:
            pass

    def _run(self) -> Iterable[Dict[str, Any]]:
        raise NotImplementedError
