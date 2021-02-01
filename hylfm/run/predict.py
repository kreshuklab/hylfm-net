from typing import Any, Dict, Iterable

from torch import no_grad
from tqdm import tqdm

from .base import Run


class PredictRun(Run):
    def __init__(self, **super_kwargs):
        super().__init__(**super_kwargs)

    @no_grad()
    def _run(self) -> Iterable[Dict[str, Any]]:
        epoch_len = len(self.dataloader)
        for it, batch in tqdm(enumerate(self.dataloader), desc=self.__class__.__name__, total=epoch_len):
            assert "epoch" not in batch
            batch["epoch"] = 0
            assert "iteration" not in batch
            batch["iteration"] = it
            assert "epoch_len" not in batch
            batch["epoch_len"] = epoch_len

            batch = self.batch_preprocessing_in_step(batch)
            batch["pred"] = self.model(batch["lfc"])
            batch = self.batch_postprocessing(batch)

            pred = batch[self.pred_name]
            tgt = batch[self.tgt_name]
            self.metrics.update_with_batch(prediction=pred, target=tgt)

            yield batch
