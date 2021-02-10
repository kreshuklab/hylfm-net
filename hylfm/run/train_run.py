import logging
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Type

import hylfm.metrics
import torch.utils.data
from tqdm import tqdm

from hylfm.checkpoint import Checkpoint
from hylfm.hylfm_types import (
    CriterionLike,
    DatasetPart,
    LRScheduler,
    LRSchedulerChoice,
    Optimizer,
    OptimizerChoice,
    TransformLike,
)
from hylfm.metrics.base import MetricGroup
from hylfm.utils.general import Period
from .base import Run
from .eval_run import ValidationRun
from .run_logger import RunLogger, WandbLogger
from .. import settings
from ..get_criterion import get_criterion
from ..get_model import get_model
from ..model import HyLFM_Net

logger = logging.getLogger(__name__)


class TrainRun(Run):
    # state
    best_validation_score: float
    impatience: int
    validation_iteration: int
    epoch: int
    iteration: int
    training_run_id: str

    def __init__(self, *, checkpoint: Checkpoint):
        cfg = checkpoint.config
        model: HyLFM_Net = get_model(**cfg.model)
        if checkpoint.model_weights is not None:
            model.load_state_dict(checkpoint.model_weights, strict=True)

        super().__init__(
            model=model,
            config=checkpoint.config,
            name=checkpoint.training_run_name,
            dataset_parts=(DatasetPart.train, DatasetPart.validate),
        )

        self.current_best_checkpoint_on_disk: Optional[Path] = None

        self.criterion = get_criterion(
            config=self.config, transforms_pipeline=self.transforms_pipelines[DatasetPart.train]
        )

        opt_class: Type[Optimizer] = getattr(torch.optim, self.config.optimizer.name)
        opt_kwargs = {"lr": self.config.opt_lr, "weight_decay": self.config.opt_weight_decay}
        if self.config.optimizer == OptimizerChoice.SGD:
            opt_kwargs["momentum"] = self.config.opt_momentum

        self.optimizer: Optimizer = opt_class(self.model.parameters(), **opt_kwargs)
        self.optimizer.zero_grad()  # calling zero_grad here, because of how batch_multiplier is implemented in TrainRun

        if checkpoint.opt_state_dict is not None:
            self.optimizer.load_state_dict(checkpoint.opt_state_dict)

        if self.config.lr_scheduler is None:
            assert checkpoint.lr_scheduler_state_dict is None
            self.lr_scheduler = None
        else:
            sched_class: Type[LRScheduler] = getattr(torch.optim.lr_scheduler, self.config.lr_scheduler.name)
            if self.config.lr_scheduler == LRSchedulerChoice.ReduceLROnPlateau:
                sched_kwargs = dict(
                    mode="min" if getattr(hylfm.metrics, cfg.score_metric.name).minimize else "max",
                    factor=cfg.lr_sched_factor,
                    patience=cfg.lr_sched_patience,
                    threshold=cfg.lr_sched_thres,
                    threshold_mode=cfg.lr_sched_thres_mode,
                    cooldown=0,
                    min_lr=1e-7,
                )
            else:
                raise NotImplementedError

            self.lr_scheduler: LRScheduler = sched_class(self.optimizer, **sched_kwargs)
            if checkpoint.lr_scheduler_state_dict is not None:
                self.lr_scheduler.load_state_dict(checkpoint.lr_scheduler_state_dict)

        self.root = settings.log_dir / "checkpoints" / self.training_run_name
        self.root.mkdir(parents=True, exist_ok=True)

        self.validator = ValidationRun(
            # batch_postprocessing=transforms_pipelines[part].batch_postprocessing,
            # batch_premetric_trf=transforms_pipelines[part].batch_premetric_trf,
            # batch_preprocessing_in_step=transforms_pipelines[part].batch_preprocessing_in_step,
            # dataloader=dataloaders[part],
            # batch_size=cfg.eval_batch_size,
            # log_pred_vs_spim=False,
            # metrics=metric_groups[part],
            # minimize=getattr(metrics, score_metric.replace("-", "_")).minimize,
            # model=model,
            # run_logger=WandbValidationLogger(
            #     point_cloud_threshold=0.3,
            #     zyx_scaling=(5, 0.7 * 8 / scale, 0.7 * 8 / scale),
            #     score_metric=score_metric,
            #     minimize=getattr(metrics, score_metric.replace("-", "_")).minimize,
            # ),
            # save_pred_to_disk=None,
            # save_spim_to_disk=None,
            # score_metric=score_metric,
            # tgt_name=transforms_pipelines[part].tgt_name,
        )
        self.validate_every = Period(cfg.validate_every_value, cfg.validate_every_unit)
        self.epoch_len = len(self.dataloaders[DatasetPart.train])

        self.batch_preprocessing_in_step = self.


        self.best_validation_score = checkpoint.best_validation_score
        self.epoch = checkpoint.epoch
        self.impatience = checkpoint.impatience
        self.iteration = checkpoint.iteration
        self.training_run_id = checkpoint.training_run_id
        self.validation_iteration = checkpoint.validation_iteration
        self.run_logger = WandbLogger(
            point_cloud_threshold=0.3, zyx_scaling=(5, 0.7 * 8 / self.scale, 0.7 * 8 / self.scale)
        )

    def save_a_checkpoint(self, best: bool, keep_anyway: bool):
        Checkpoint(
            best_validation_score=self.best_validation_score,
            config=self.config,
            epoch=self.epoch,
            impatience=self.impatience,
            iteration=self.iteration,
            model_weights=self.model.state_dict(),
            opt_state_dict=self.optimizer.state_dict(),
            scheduler_state_dict=self.scheduler.state_dict(),
            training_run_id=self.training_run_id,
            training_run_name=self.name,
            validation_iteration=self.validation_iteration,
        ).save(best=best, keep_anyway=keep_anyway)

    def fit(self):
        for _ in self:
            pass

    def _validate(self) -> float:
        self.validation_iteration += 1
        validation_score = self.validator.get_validation_score(
            step=(self.epoch * self.epoch_len + self.iteration) * self.batch_size
        )
        best = self.best_validation_score is None or self.best_validation_score < validation_score
        val_it_is_power_of_two = self.validation_iteration & (self.validation_iteration - 1) == 0
        if val_it_is_power_of_two or best:
            self.save_a_checkpoint(best=best, keep_anyway=val_it_is_power_of_two)

        if best:
            self.best_validation_score = validation_score
            self.impatience = 0
        else:
            self.impatience += 1

        self.model.train()
        return validation_score

    def _step(self, batch: dict):
        ep = self.epoch
        it = self.iteration

        assert "epoch" not in batch
        batch["epoch"] = ep
        assert "iteration" not in batch
        batch["iteration"] = it
        assert "epoch_len" not in batch
        batch["epoch_len"] = self.epoch_len

        assert "batch_len" in batch

        batch = self.batch_preprocessing_in_step(batch)
        batch["pred"] = self.model(batch["lfc"])
        batch = self.batch_postprocessing(batch)

        loss = (
            self.criterion(batch["pred"], batch[self.tgt_name], epoch=ep, iteration=it, epoch_len=self.epoch_len)
            / self.batch_multiplier
        )
        if not self.criterion.minimize:
            loss *= -1

        loss.backward()
        if (it + 1) % self.batch_multiplier == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()

        batch = self.batch_premetric_trf(batch)
        step_metrics = self.train_metrics.update_with_batch(prediction=batch["pred"], target=batch[self.tgt_name])
        step_metrics[self.criterion.__class__.__name__ + "_loss"] = loss.item()
        for from_batch in ["NormalizeMSE.alpha", "NormalizeMSE.beta"]:
            assert from_batch not in step_metrics
            if from_batch in batch:
                step_metrics[from_batch] = batch[from_batch]

        opt_param_groups = self.optimizer.state_dict()["param_groups"]
        step_metrics["lr"] = opt_param_groups[0]["lr"]

        if self.validate_every.match(epoch=ep, iteration=it, epoch_len=self.epoch_len):
            step_metrics[self.validator.score_metric + "_val-score"] = self._validate()

        step = self.epoch * self.epoch_len + self.iteration
        self.run_logger(epoch=ep, iteration=it, epoch_len=self.epoch_len, step=step, **step_metrics)

        return batch

    def _run(self) -> Iterable[Dict[str, Any]]:
        self.model.train()
        stop_early = False
        zero_max_threshold = 0.01
        zero_max_patience = 10
        zero_max_impatience = 0
        for epoch in range(self.epoch, self.max_epochs):
            self.epoch = epoch
            for it, batch in tqdm(
                enumerate(self.dataloader), desc=f"{self.name}|ep {epoch + 1:3}/{self.max_epochs}", total=self.epoch_len
            ):
                if it < self.iteration:
                    # catch up with loaded state
                    logger.warning("skipping iteration %s to resume at %s", it, self.iteration)
                    continue

                if self.impatience > self.patience:
                    stop_early = True
                    logger.warning("stopping early after %s non-improving validations", self.impatience)
                    break

                self.iteration = it
                batch = self._step(batch)
                yield batch
                if batch["pred"].max() < zero_max_threshold:
                    zero_max_impatience += 1
                    if zero_max_impatience > zero_max_patience:
                        self.iteration += 1
                        stop_early = True
                        logger.warning(
                            "stopping early after %s consecutive pred.max() < %s",
                            zero_max_impatience,
                            zero_max_threshold,
                        )
                        break

            if stop_early:
                break

            self.iteration = 0
        else:
            self.epoch += 1

        self.run_logger.log_summary(step=self.epoch * self.epoch_len + self.iteration, **self.metrics.compute())
        self.metrics.reset()
