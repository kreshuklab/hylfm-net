from typing import Any, Callable, Dict, Iterable

import torch
from torch import no_grad
from torch.optim import Optimizer
from tqdm import tqdm

from .base import Run
from .eval import EvalRun, ValidationRun
from hylfm.hylfm_types import CriterionLike
from hylfm.metrics.base import MetricGroup
from hylfm.utils.general import Period, PeriodUnit


class TrainRun(Run):
    def __init__(
        self,
        *,
        max_epochs: int,
        train_metrics: MetricGroup,
        validator: ValidationRun,
        validate_every: Period,
        patience: int,
        criterion: CriterionLike,
        batch_multiplier: int = 1,
        optimizer: Optimizer,
        **super_kwargs
    ):
        assert max_epochs > 0, max_epochs
        super().__init__(**super_kwargs)
        self.max_epochs = max_epochs
        self.train_metrics = train_metrics
        self.validator = validator
        self.vadate_every = validate_every
        self.patience = patience
        self.criterion = criterion
        self.batch_multiplier = batch_multiplier
        self.optimizer = optimizer

    def fit(self):
        for it in self:
            pass

    def _run(self) -> Iterable[Dict[str, Any]]:
        self.model.train()
        epoch_len = len(self.dataloader)
        assert epoch_len
        impatience = 0
        best_validation_score = None
        stop_early = False
        for epoch in range(self.max_epochs):
            for it, batch in tqdm(enumerate(self.dataloader), desc=self.name, total=epoch_len):
                if impatience > self.patience:
                    stop_early = True
                    break

                assert "epoch" not in batch
                batch["epoch"] = 0
                assert "iteration" not in batch
                batch["iteration"] = it
                assert "epoch_len" not in batch
                batch["epoch_len"] = epoch_len

                batch = self.batch_preprocessing_in_step(batch)
                batch["pred"] = self.model(batch["lfc"])
                batch = self.batch_postprocessing(batch)

                loss = self.criterion(batch["pred"], batch[self.tgt_name]) / self.batch_multiplier
                loss.backward()
                if (it + 1) % self.batch_multiplier == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                batch = self.batch_premetric_trf(batch)
                step_metrics = self.train_metrics.update_with_batch(
                    prediction=batch["pred"], target=batch[self.tgt_name]
                )
                step_metrics[self.criterion.__class__.__name__] = loss.item()

                if self.vadate_every.match(epoch=epoch, iteration=it, epoch_len=epoch_len):
                    validation_score = self.validator.get_validation_score()
                    self.model.train()
                    step_metrics[self.validator.score_metric + "_val-score"] = validation_score
                    if best_validation_score is None or best_validation_score < validation_score:
                        best_validation_score = validation_score
                        impatience = 0
                    else:
                        impatience += 1

                self.run_logger(
                    epoch=epoch, epoch_len=epoch_len, iteration=it, batch_len=batch["batch_len"], **step_metrics
                )

                yield batch

            if stop_early:
                break

        self.run_logger.log_summary(**self.metrics.compute())
        self.metrics.reset()
