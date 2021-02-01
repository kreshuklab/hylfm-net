from typing import Any, Dict, Iterable

from torch import no_grad
from tqdm import tqdm

from .base import Run
from ..metrics.base import MetricGroup


class TrainRun(Run):
    def __init__(self, *, train_metrics: MetricGroup, **super_kwargs):
        super().__init__(**super_kwargs)
        self.train_metrics = train_metrics

    @no_grad()
    def _run(self) -> Iterable[Dict[str, Any]]:
        epoch_len = len(self.dataloader)
        for it, batch in tqdm(enumerate(self.dataloader), desc=self.name, total=epoch_len):
            assert "epoch" not in batch
            batch["epoch"] = 0
            assert "iteration" not in batch
            batch["iteration"] = it
            assert "epoch_len" not in batch
            batch["epoch_len"] = epoch_len

            batch = self.batch_preprocessing_in_step(batch)
            pred = self.model(batch["lfc"])
            batch["pred"] = pred
            batch = self.batch_postprocessing(batch)

            engine.state.criterion(tensors)
            loss = tensors[stage.criterion_setup.name] / stage.batch_multiplier
            loss.backward()
            if (engine.state.iteration + 1) % stage.batch_multiplier == 0:
                engine.state.optimizer.step()
                engine.state.optimizer.zero_grad()

            if self.tgt_name is not None:
                tgt = batch[self.tgt_name]
                self.train_metrics.update_with_batch(prediction=pred, target=tgt)

            yield batch
