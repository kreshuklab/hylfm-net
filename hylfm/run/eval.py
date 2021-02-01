from typing import Any, Dict, Iterable

from torch import no_grad
from tqdm import tqdm

from .base import Run


class EvalRun(Run):
    @no_grad()
    def _run(self) -> Iterable[Dict[str, Any]]:
        epoch_len = len(self.dataloader)
        assert epoch_len
        for it, batch in tqdm(enumerate(self.dataloader), desc=self.name, total=epoch_len):
            assert "epoch" not in batch
            batch["epoch"] = 0
            assert "iteration" not in batch
            batch["iteration"] = it
            assert "epoch_len" not in batch
            batch["epoch_len"] = epoch_len

            batch = self.batch_preprocessing_in_step(batch)
            batch["pred"] = self.model(batch["lfc"])
            batch = self.batch_postprocessing(batch)
            if self.tgt_name is not None:
                batch = self.batch_premetric_trf(batch)
                self.metrics.update_with_batch(prediction=batch["pred"], target=batch[self.tgt_name])

            yield batch

        self.log_run(**batch)  # noqa
        self.metrics.reset()
