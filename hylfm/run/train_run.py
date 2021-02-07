from typing import Any, Dict, Iterable, Optional

import torch.utils.data
from torch.optim import Optimizer
from tqdm import tqdm

from hylfm.checkpoint import Checkpoint
from hylfm.hylfm_types import CriterionLike, TransformLike
from hylfm.metrics.base import MetricGroup
from hylfm.utils.general import Period
from .base import Run
from .eval import ValidationRun
from .run_logger import RunLogger


class TrainRun(Run):
    # state
    best_validation_score: float
    impatience: int
    last_batch_len: Optional[int]
    validation_iteration: int
    epoch: int
    iteration: int
    training_run_id: str

    def __init__(
        self,
        *,
        batch_postprocessing: TransformLike,
        batch_premetric_trf: TransformLike,
        batch_preprocessing_in_step: TransformLike,
        criterion: CriterionLike,
        dataloader: torch.utils.data.DataLoader,
        metrics: MetricGroup,
        model: torch.nn.Module,
        optimizer: Optimizer,
        pred_name: str,
        run_logger: RunLogger,
        tgt_name: Optional[str],
        train_metrics: MetricGroup,
        validator: ValidationRun,
        checkpoint: Checkpoint,
    ):
        super().__init__(
            model=model,
            dataloader=dataloader,
            batch_preprocessing_in_step=batch_preprocessing_in_step,
            batch_postprocessing=batch_postprocessing,
            batch_premetric_trf=batch_premetric_trf,
            metrics=metrics,
            pred_name=pred_name,
            tgt_name=tgt_name,
            run_logger=run_logger,
            name=checkpoint.training_run_name,
        )
        self.train_metrics = train_metrics
        self.validator = validator
        self.criterion = criterion
        self.optimizer = optimizer

        cfg = checkpoint.config
        self.config_to_save_with_checkpoint = cfg
        self.validate_every = Period(cfg.validate_every_value, cfg.validate_every_unit)
        self.max_epochs = cfg.max_epochs
        self.patience = cfg.patience
        self.batch_multiplier = cfg.batch_multiplier

        self.set_state(checkpoint)
        self.epoch_len = len(self.dataloader)
        assert self.epoch_len

    def set_state(self, checkpoint: Checkpoint):
        self.best_validation_score = checkpoint.best_validation_score
        self.epoch = checkpoint.epoch
        self.impatience = checkpoint.impatience
        self.iteration = checkpoint.iteration
        self.last_batch_len = checkpoint.last_batch_len
        self.training_run_id = checkpoint.training_run_id
        self.validation_iteration = checkpoint.validation_iteration

    def save_a_checkpoint(self, best: bool, keep_anyway: bool):
        Checkpoint(
            best_validation_score=self.best_validation_score,
            config=self.config_to_save_with_checkpoint,
            epoch=self.epoch,
            impatience=self.impatience,
            iteration=self.iteration,
            last_batch_len=self.last_batch_len,
            model_weights=self.model.state_dict(),
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
            step=(self.epoch * self.epoch_len + self.iteration) * (self.last_batch_len or self.current_batch_len)
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
        self.current_batch_len: int = batch["batch_len"]

        batch = self.batch_preprocessing_in_step(batch)
        batch["pred"] = self.model(batch["lfc"])
        batch = self.batch_postprocessing(batch)

        loss = self.criterion(batch["pred"], batch[self.tgt_name]) / self.batch_multiplier
        if not self.criterion.minimize:
            loss *= -1

        loss.backward()
        if (it + 1) % self.batch_multiplier == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()

        batch = self.batch_premetric_trf(batch)
        step_metrics = self.train_metrics.update_with_batch(prediction=batch["pred"], target=batch[self.tgt_name])
        step_metrics[self.criterion.__class__.__name__ + "_loss"] = loss.item()

        if self.validate_every.match(epoch=ep, iteration=it, epoch_len=self.epoch_len):
            step_metrics[self.validator.score_metric + "_val-score"] = self._validate()

        self.run_logger(epoch=ep, epoch_len=self.epoch_len, iteration=it, batch_len=batch["batch_len"], **step_metrics)

        self.last_batch_len = batch["batch_len"]
        return batch

    def _run(self) -> Iterable[Dict[str, Any]]:
        self.model.train()
        stop_early = False
        for epoch in range(self.epoch, self.max_epochs):
            self.epoch = epoch
            for it, batch in tqdm(
                enumerate(self.dataloader), desc=f"{self.name}|ep {epoch:3}/{self.max_epochs}", total=self.epoch_len
            ):
                if it < self.iteration:
                    continue  # catch up with loaded state

                if self.impatience > self.patience:
                    stop_early = True
                    break

                self.iteration = it
                yield self._step(batch)

            if stop_early:
                break

        self.run_logger.log_summary(**self.metrics.compute())
        self.metrics.reset()
