from typing import Any, Dict, Iterable

import numpy
import torch
from torch import no_grad
from tqdm import tqdm

from .base import Run


class EvalRun(Run):
    @no_grad()
    def _run(self) -> Iterable[Dict[str, Any]]:
        epoch = 0
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
                step_metrics = self.metrics.update_with_batch(prediction=batch["pred"], target=batch[self.tgt_name])

                # step_metrics["prediction"] = list(batch["pred"])
                # step_metrics["spim"] = list(batch[self.tgt_name])

                pred = batch["pred"]
                spim = batch[self.tgt_name]

                # zeros = torch.zeros_like(pred)
                # pred = torch.cat([zeros, pred, pred], dim=1)
                # spim = torch.cat([spim, zeros, spim], dim=1)
                # step_metrics["pred-vs-spim"] = list(pred + spim)

                step_metrics["pred-vs-spim"] = list(torch.cat([pred, spim], dim=1))

                if self.run_logger is not None:
                    self.run_logger(
                        epoch=epoch, epoch_len=epoch_len, iteration=it, batch_len=batch["batch_len"], **step_metrics
                    )

            yield batch

        self.run_logger.log_summary(**self.metrics.compute())
        self.metrics.reset()
