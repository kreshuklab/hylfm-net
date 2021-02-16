import logging
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Type

import torch.utils.data
from tqdm import tqdm

import hylfm.metrics
from hylfm.checkpoint import Checkpoint, TrainRunConfig, ValidationRunConfig
from hylfm.get_criterion import get_criterion
from hylfm.get_model import get_model
from hylfm.hylfm_types import DatasetPart, LRScheduler, LRSchedulerChoice, Optimizer, OptimizerChoice
from hylfm.model import HyLFM_Net
from hylfm.utils.general import Period
from .base import Run
from .eval_run import ValidationRun
from .run_logger import WandbLogger

logger = logging.getLogger(__name__)


class TrainRun(Run):
    config: TrainRunConfig

    def __init__(self, *, wandb_run, checkpoint: Checkpoint):
        cfg = checkpoint.config
        model: HyLFM_Net = get_model(**cfg.model)
        if checkpoint.model_weights is not None:
            model.load_state_dict(checkpoint.model_weights, strict=True)

        self.wandb_run = wandb_run
        assert wandb_run.name == checkpoint.training_run_name
        scale = model.get_scale()
        super().__init__(
            config=checkpoint.config,
            dataset_part=DatasetPart.train,
            model=model,
            name=checkpoint.training_run_name,
            run_logger=WandbLogger(
                point_cloud_threshold=cfg.point_cloud_threshold, zyx_scaling=(5, 0.7 * 8 / scale, 0.7 * 8 / scale)
            ),
        )

        self.current_best_checkpoint_on_disk: Optional[Path] = None

        self.criterion = get_criterion(config=self.config, transforms_pipeline=self.transforms_pipeline)

        opt_class: Type[Optimizer] = getattr(torch.optim, self.config.optimizer.name)
        opt_kwargs = {"lr": self.config.opt_lr, "weight_decay": self.config.opt_weight_decay}
        if self.config.optimizer == OptimizerChoice.SGD:
            opt_kwargs["momentum"] = self.config.opt_momentum

        self.optimizer: Optimizer = opt_class(self.model.parameters(), **opt_kwargs)
        self.optimizer.zero_grad()  # calling zero_grad here, because of how batch_multiplier is implemented in TrainRun

        if checkpoint.optimizer_state_dict is not None:
            self.optimizer.load_state_dict(checkpoint.optimizer_state_dict)

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

        self.validator = ValidationRun(
            config=ValidationRunConfig(
                batch_size=cfg.eval_batch_size,
                data_range=cfg.data_range,
                dataset=cfg.dataset,
                interpolation_order=cfg.interpolation_order,
                save_output_to_disk={},
                win_sigma=cfg.win_sigma,
                win_size=cfg.win_size,
                hylfm_version=cfg.hylfm_version,
                point_cloud_threshold=cfg.point_cloud_threshold,
            ),
            model=model,
            score_metric=cfg.score_metric,
            name=self.name,
        )
        self.validate_every = Period(cfg.validate_every_value, cfg.validate_every_unit)
        self.epoch_len = len(self.dataloader)

        self.best_validation_score = checkpoint.best_validation_score
        self.epoch = checkpoint.epoch
        self.impatience = checkpoint.impatience
        self.iteration = checkpoint.iteration
        self.training_run_id = checkpoint.training_run_id
        self.validation_iteration = checkpoint.validation_iteration

    def get_checkpoint(self):
        return Checkpoint(
            config=self.config,
            training_run_id=self.training_run_id,
            training_run_name=self.name,
            best_validation_score=self.best_validation_score,
            epoch=self.epoch,
            impatience=self.impatience,
            iteration=self.iteration,
            model_weights=self.model.state_dict(),
            optimizer_state_dict=self.optimizer.state_dict(),
            lr_scheduler_state_dict=None if self.lr_scheduler is None else self.lr_scheduler.state_dict(),
            validation_iteration=self.validation_iteration,
        )

    def save_a_checkpoint(self, best: bool, keep_anyway: bool):
        if not (best or keep_anyway):
            return

        if best and self.current_best_checkpoint_on_disk is not None:
            # remove old best
            try:
                self.current_best_checkpoint_on_disk.unlink()
            except Exception as e:
                logger.warning(
                    "Could not remove old best checkpoint %s, due to %s", self.current_best_checkpoint_on_disk, e
                )

        path = self.get_checkpoint().save(best=best)

        # remember current best to delete on finding new best
        self.current_best_checkpoint_on_disk = None if keep_anyway else path

    def fit(self):
        for _ in self:
            pass

    def _validate(self, last: bool = False) -> float:
        self.validation_iteration += 1
        validation_score = self.validator.get_validation_score(
            step=(self.epoch * self.epoch_len + self.iteration) * self.config.batch_size
        )
        self.model.train()

        best = self.best_validation_score is None or self.best_validation_score < validation_score
        val_it_is_power_of_two = self.validation_iteration & (self.validation_iteration - 1) == 0

        if best:
            self.best_validation_score = validation_score
            self.impatience = 0
        else:
            self.impatience += 1

        keep_anyway = (
            last or val_it_is_power_of_two or self.validation_iteration in self.config.save_after_validation_iterations
        )
        if keep_anyway or best:
            self.save_a_checkpoint(best=best, keep_anyway=keep_anyway)

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

        batch = self.transforms_pipeline.batch_preprocessing_in_step(batch)
        batch["pred"] = self.model(batch["lfc"])
        batch = self.transforms_pipeline.batch_postprocessing(batch)

        loss = (
            self.criterion(
                batch["pred"],
                batch[self.transforms_pipeline.tgt_name],
                epoch=ep,
                iteration=it,
                epoch_len=self.epoch_len,
            )
            / self.config.batch_multiplier
        )
        if not self.criterion.minimize:
            loss *= -1

        loss.backward()
        if (it + 1) % self.config.batch_multiplier == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()

        batch = self.transforms_pipeline.batch_premetric_trf(batch)
        step_metrics = self.metric_group.update_with_batch(
            prediction=batch["pred"], target=batch[self.transforms_pipeline.tgt_name]
        )
        step_metrics[self.criterion.__class__.__name__ + "_loss"] = loss.item()
        for from_batch in ["NormalizeMSE.alpha", "NormalizeMSE.beta"]:
            assert from_batch not in step_metrics
            if from_batch in batch:
                step_metrics[from_batch] = batch[from_batch]

        opt_param_groups = self.optimizer.state_dict()["param_groups"]

        if self.validate_every.match(epoch=ep, iteration=it, epoch_len=self.epoch_len):
            validation_score = self._validate()

            step_metrics[self.validator.score_metric + "_val-score"] = validation_score
            if self.lr_scheduler is not None:
                self.lr_scheduler.step(validation_score)

        step_metrics["lr"] = opt_param_groups[0]["lr"]
        step = (ep * self.epoch_len + it) * self.config.batch_size
        self.run_logger(epoch=ep, iteration=it, epoch_len=self.epoch_len, step=step, **step_metrics)

        return batch

    def _run(self) -> Iterable[Dict[str, Any]]:
        self.model.train()
        stop_early = False
        zero_max_threshold = 0.01
        zero_max_patience = 10
        zero_max_impatience = 0

        if self.iteration + 1 >= self.epoch_len:
            self.epoch += 1
            self.iteration = 0

        catch_up_to_iteration = self.iteration

        for epoch in range(self.epoch, self.config.max_epochs):
            self.epoch = epoch
            for it, batch in tqdm(
                enumerate(self.dataloader),
                desc=f"{self.name}|ep {epoch + 1:3}/{self.config.max_epochs}",
                total=self.epoch_len,
            ):
                if it < catch_up_to_iteration:
                    # catch up with loaded state  # todo: improve speed by not loading batch data in this case
                    logger.warning("skipping iteration %s to resume at %s", it, self.iteration)
                    continue

                self.iteration = it
                batch = self._step(batch)

                yield batch

                if self.impatience > self.config.patience:
                    stop_early = True
                    logger.warning("stopping early after %s non-improving validations", self.impatience)
                    break

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

        if not stop_early:
            self._validate(last=True)

        self.run_logger.log_summary(step=self.epoch * self.epoch_len + self.iteration, **self.metric_group.compute())
        self.metric_group.reset()
