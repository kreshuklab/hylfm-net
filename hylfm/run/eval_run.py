import collections
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import pandas
import torch
from torch import no_grad
from tqdm import tqdm

import hylfm.metrics
from hylfm.checkpoint import (
    PredictPathRunConfig,
    RunConfig,
    TestCheckpointRunConfig,
    TestPrecomputedRunConfig,
    ValidationRunConfig,
)
from hylfm.datasets import ConcatDataset, TensorInfo, ZipDataset, get_dataset_from_info
from hylfm.get_model import get_model
from hylfm.hylfm_types import DatasetChoice, DatasetPart, MetricChoice
from hylfm.model import HyLFM_Net
from hylfm.utils.for_log import get_max_projection_img
from hylfm.utils.io import save_pandas_df, save_tensor
from .base import Run
from .run_logger import WandbLogger, WandbValidationLogger
from ..datasets.named import get_dataset
from ..transform_pipelines import get_transforms_pipeline


@dataclass
class EvalYield:
    batch: Optional[Dict[str, Any]] = None
    step_metrics: Optional[Dict[str, Any]] = None
    summary_metrics: Optional[Dict[str, Any]] = None


class EvalRun(Run):
    def __init__(
        self,
        *,
        log_level_wandb: int,
        config: RunConfig,
        model: Optional[HyLFM_Net],
        dataset_part: DatasetPart,
        name: str,
        run_logger: WandbLogger,
        scale: Optional[int] = None,
        shrink: Optional[int] = None,
    ):
        super().__init__(
            config=config,
            model=model,
            dataset_part=dataset_part,
            name=name,
            run_logger=run_logger,
            scale=scale,
            shrink=shrink,
        )
        self.log_level_wandb = log_level_wandb

    @staticmethod
    def progress_tqdm(iterable, desc: str, total: int):
        return tqdm(iterable, desc=desc, total=total)

    @no_grad()
    def get_pred(self, batch):
        return self.model(batch["lfc"])

    @no_grad()
    def _run(self) -> Iterable[EvalYield]:
        trfs = self.transforms_pipeline
        if self.model is not None:
            self.model.eval()

        epoch = 0
        it = 0
        tab_data_per_step = collections.defaultdict(list)
        sample_idx = 0

        def save_tensor_batch(root: Path, tensor_batch):
            for batch_idx, tensor in enumerate(tensor_batch):
                file_path = root / f"{sample_idx + batch_idx:05}.tif"
                save_tensor(file_path, tensor)

        for it, batch in self.progress_tqdm(enumerate(self.dataloader), desc=self.name, total=self.epoch_len):
            assert "epoch" not in batch
            batch["epoch"] = 0
            assert "iteration" not in batch
            batch["iteration"] = it
            assert "epoch_len" not in batch
            batch["epoch_len"] = self.epoch_len

            batch = trfs.batch_preprocessing_in_step(batch)
            batch["pred"] = self.get_pred(batch)
            batch = trfs.batch_postprocessing(batch)
            batch = trfs.batch_premetric_trf(batch)

            if trfs.tgt_name is None:
                step_metrics = {}
            else:
                step_metrics = self.metric_group.update_with_batch(
                    prediction=batch["pred"], target=batch[trfs.tgt_name]
                )

            for from_batch in ["NormalizeMSE.alpha", "NormalizeMSE.beta"]:
                assert from_batch not in step_metrics
                if from_batch in batch:
                    step_metrics[from_batch] = batch[from_batch]

                    # color version does not work somehow...
                    # zeros = torch.zeros_like(pred)
                    # pred = torch.cat([zeros, pred, pred], dim=1)
                    # spim = torch.cat([spim, zeros, spim], dim=1)
                    # step_metrics["pred-vs-spim"] = list(pred + spim)

            if "metrics" in self.save_output_to_disk:
                for key, val in step_metrics.items():
                    tab_data_per_step[key].append(val)

            if self.log_level_wandb > 0:
                pred = batch["pred"]
                pr = pred.detach().cpu().numpy()
                pr = get_max_projection_img(pr)
                step_metrics["pred_max"] = list(pr)

                if self.log_level_wandb > 1:
                    step_metrics["pred-cloud"] = list(pr)

                if self.log_level_wandb > 2:
                    if trfs.tgt_name in batch:
                        spim = batch[trfs.tgt_name]
                        sp = spim.detach().cpu().numpy()
                        step_metrics["spim_max"] = list(sp.max(2))
                        step_metrics["pred-vs-spim"] = list(torch.cat([pred, spim], dim=1))

            if self.log_level_wandb > 3:
                lf = batch["lf"]
                assert len(lf.shape) == 4, lf.shape
                step_metrics["lf"] = list(lf)

            step = (epoch * self.epoch_len + it) * self.config.batch_size
            self.run_logger(epoch=epoch, iteration=it, epoch_len=self.epoch_len, step=step, **step_metrics)

            for key, path in self.save_output_to_disk.items():
                if key == "metrics":
                    continue  # added to tab_data_per_step above
                elif key not in batch:
                    if key == "spim" and "ls_slice" in batch and "ls_slice" not in self.save_output_to_disk:
                        key = "ls_slice"
                    elif key == "spim" and "ls_trf" in batch and "ls_trf" not in self.save_output_to_disk:
                        key = "ls_trf"
                    elif key == "spim" and "lfd" in batch and "lfd" not in self.save_output_to_disk:
                        key = "lfd"  # train on lfd, todo: rename 'spim' to 'target'
                    else:
                        raise NotImplementedError(key)

                save_tensor_batch(path, batch[key])

            sample_idx += batch["batch_len"]
            yield EvalYield(batch=batch, step_metrics=step_metrics)

        summary_metrics = self.metric_group.compute()
        if "metrics" in self.save_output_to_disk:
            df = pandas.DataFrame.from_dict(tab_data_per_step)
            df_path = self.save_output_to_disk["metrics"]
            save_pandas_df(df, df_path)

        self.run_logger.log_summary(
            step=(epoch * self.epoch_len + it + 1) * self.config.batch_size - 1, **summary_metrics
        )
        self.metric_group.reset()
        yield EvalYield(summary_metrics=summary_metrics)


class ValidationRun(EvalRun):
    def __init__(self, *, config: ValidationRunConfig, model: HyLFM_Net, score_metric: MetricChoice, name: str):
        scale = model.get_scale()
        self.minimize = getattr(hylfm.metrics, score_metric.replace("-", "_")).minimize
        super().__init__(
            config=config,
            model=model,
            dataset_part=DatasetPart.validate,
            log_level_wandb=0,
            name=name,
            run_logger=WandbValidationLogger(
                point_cloud_threshold=config.point_cloud_threshold,
                zyx_scaling=(5, 0.7 * 8 / scale, 0.7 * 8 / scale),
                score_metric=score_metric,
                minimize=self.minimize,
            ),
        )
        self.score_metric = score_metric

    @staticmethod
    def progress_tqdm(iterable, **kwargs):
        # do not log progress of validation
        return iterable

    @no_grad()
    def get_validation_score(self, step: int) -> float:
        self.run_logger.step = step
        summary = None
        for y in self:
            summary = y.summary_metrics

        score = summary[self.score_metric]
        if self.minimize:
            score *= -1

        return score


class TestCheckpointRun(EvalRun):
    def __init__(self, wandb_run, config: TestCheckpointRunConfig, log_level_wandb: int):
        model: HyLFM_Net = get_model(**config.checkpoint.config.model)
        model.load_state_dict(config.checkpoint.model_weights, strict=True)

        self.wandb_run = wandb_run
        scale = model.get_scale()
        super().__init__(
            config=config,
            dataset_part=DatasetPart.test,
            model=model,
            name=config.checkpoint.training_run_name,
            run_logger=WandbLogger(
                point_cloud_threshold=config.point_cloud_threshold, zyx_scaling=(5, 0.7 * 8 / scale, 0.7 * 8 / scale)
            ),
            log_level_wandb=log_level_wandb,
        )


class TestPrecomputedRun(EvalRun):
    config: TestPrecomputedRunConfig

    def __init__(self, *, wandb_run, config: TestPrecomputedRunConfig, scale: int, shrink: int, log_level_wandb: int):
        self.load_lfd_and_care = config.pred_name in ["lfd", "care"] or config.trgt_name in ["lfd", "care"]
        super().__init__(
            config=config,
            model=None,
            dataset_part=DatasetPart.test,
            name=wandb_run.name,
            run_logger=WandbLogger(
                point_cloud_threshold=config.point_cloud_threshold, zyx_scaling=(5, 0.7 * 8 / scale, 0.7 * 8 / scale)
            ),
            log_level_wandb=log_level_wandb,
            scale=scale,
            shrink=shrink,
        )

    def get_pred(self, batch):
        assert self.config.pred_name is not None
        if self.shrink:
            return batch[self.config.pred_name][..., self.shrink : -self.shrink, self.shrink : -self.shrink]

    def get_dataset(self):
        if self.config.dataset == DatasetChoice.from_path:
            assert self.dataset_part == DatasetPart.test

            tensor_infos = {
                self.config.pred_name: TensorInfo(
                    name=self.config.pred_name,
                    root=self.config.path,
                    location=self.config.pred_glob,
                    transforms=self.transforms_pipeline.sample_precache_trf,
                    datasets_per_file=1,
                    samples_per_dataset=1,
                    remove_singleton_axes_at=tuple(),  # (-1,),
                    insert_singleton_axes_at=(0, 0),
                    z_slice=None,
                    skip_indices=tuple(),
                    meta=None,
                ),
                self.config.trgt_name: TensorInfo(
                    name=self.config.pred_name,
                    root=self.config.path,
                    location=self.config.trgt_glob,
                    transforms=self.transforms_pipeline.sample_precache_trf,
                    datasets_per_file=1,
                    samples_per_dataset=1,
                    remove_singleton_axes_at=tuple(),  # (-1,),
                    insert_singleton_axes_at=(0, 0),
                    z_slice=None,
                    skip_indices=tuple(),
                    meta=None,
                ),
            }
            dtst = ZipDataset(
                {
                    name: get_dataset_from_info(ti, cache=True, filters=[], indices=None)
                    for name, ti in tensor_infos.items()
                }
            )
            return ConcatDataset([dtst], transform=self.transforms_pipeline.sample_preprocessing)

        else:
            return get_dataset(
                self.config.dataset,
                self.dataset_part,
                nnum=19,
                z_out=49,
                scale=self.scale,
                shrink=self.shrink,
                interpolation_order=self.config.interpolation_order,
                incl_pred_vol="pred_vol" in self.save_output_to_disk,
                load_lfd_and_care=self.load_lfd_and_care,
            )

    def get_transforms_pipeline(self):
        return get_transforms_pipeline(
            dataset_name=self.config.dataset,
            dataset_part=self.dataset_part,
            nnum=19 if self.model is None else self.model.nnum,
            z_out=49 if self.model is None else self.model.z_out,
            scale=self.scale,
            shrink=self.shrink,
            interpolation_order=self.config.interpolation_order,
            incl_pred_vol="pred_vol" in self.save_output_to_disk,
            load_lfd_and_care=self.load_lfd_and_care
            or "lfd" in self.save_output_to_disk
            or "care" in self.save_output_to_disk,
            tgt_name_for_from_path="spim",
        )


class PredictPathRun(EvalRun):
    config: PredictPathRunConfig

    def __init__(self, wandb_run, config: PredictPathRunConfig, log_level_wandb: int):
        model: HyLFM_Net = get_model(**config.checkpoint.config.model)
        model.load_state_dict(config.checkpoint.model_weights, strict=True)

        self.wandb_run = wandb_run
        scale = model.get_scale()
        super().__init__(
            config=config,
            dataset_part=DatasetPart.predict,
            model=model,
            name=config.checkpoint.training_run_name,
            run_logger=WandbLogger(
                point_cloud_threshold=config.point_cloud_threshold, zyx_scaling=(5, 0.7 * 8 / scale, 0.7 * 8 / scale)
            ),
            log_level_wandb=log_level_wandb,
        )

    def get_dataset(self):
        assert self.config.dataset == DatasetChoice.predict_path
        assert self.dataset_part == DatasetPart.predict

        tensor_info = TensorInfo(
            name="lf",
            root=self.config.path,
            location=self.config.glob_lf,
            transforms=self.transforms_pipeline.sample_precache_trf,
            datasets_per_file=1,
            samples_per_dataset=1,
            remove_singleton_axes_at=tuple(),  # (-1,),
            insert_singleton_axes_at=(0, 0),
            z_slice=None,
            skip_indices=tuple(),
            meta=None,
        )

        dtst = get_dataset_from_info(tensor_info, cache=True, filters=[], indices=None)
        return ConcatDataset([dtst], transform=self.transforms_pipeline.sample_preprocessing)
