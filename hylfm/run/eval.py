import collections
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import pandas
import torch
from torch import no_grad
from tqdm import tqdm

from .base import Run
from ..utils.io import save_tensor


@dataclass
class EvalYield:
    batch: Optional[Dict[str, Any]] = None
    step_metrics: Optional[Dict[str, Any]] = None
    summary_metrics: Optional[Dict[str, Any]] = None


class EvalRun(Run):
    def __init__(
        self,
        *,
        log_pred_vs_spim: bool,
        save_pred_to_disk: Optional[Path] = None,
        save_spim_to_disk: Optional[Path] = None,
        **super_kwargs,
    ):
        super().__init__(**super_kwargs)
        self.log_pred_vs_spim = log_pred_vs_spim

        self.save_pred_to_disk = save_pred_to_disk
        if save_pred_to_disk:
            assert save_spim_to_disk is None or save_spim_to_disk.name != save_pred_to_disk.name, "name used as key"
            save_pred_to_disk.mkdir(parents=True, exist_ok=True)

        self.save_spim_to_disk = save_spim_to_disk
        if save_spim_to_disk:
            assert save_pred_to_disk is None or save_pred_to_disk.name != save_spim_to_disk.name, "name used as key"
            save_spim_to_disk.mkdir(parents=True, exist_ok=True)

    @no_grad()
    def _run(self) -> Iterable[EvalYield]:
        self.model.eval()
        epoch = 0
        epoch_len = len(self.dataloader)
        assert epoch_len
        assert epoch_len < 100000 or not self.save_spim_to_disk and not self.save_pred_to_disk
        tab_data_per_step = collections.defaultdict(list)
        sample_idx = 0

        def save_tensor_batch(root: Path, tensor_batch):
            for batch_idx, tensor in enumerate(tensor_batch):
                path = root / f"{sample_idx + batch_idx:05}.tif"
                save_tensor(path, tensor)
                tab_data_per_step[root.name].append(str(path))

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
            if self.tgt_name is None:
                step_metrics = None
            else:
                batch = self.batch_premetric_trf(batch)

                step_metrics = self.metrics.update_with_batch(prediction=batch["pred"], target=batch[self.tgt_name])
                for from_batch in ["NormalizeMSE.alpha", "NormalizeMSE.beta"]:
                    assert from_batch not in step_metrics
                    if from_batch in batch:
                        step_metrics[from_batch] = batch[from_batch]

                if self.log_pred_vs_spim:
                    pred = batch["pred"]
                    spim = batch[self.tgt_name]

                    pr = pred.detach().cpu().numpy()
                    sp = spim.detach().cpu().numpy()
                    assert len(pr.shape) == 5, pr.shape
                    assert pr.shape[0] == 1, pr.shape
                    assert pr.shape[1] == 1, pr.shape
                    assert pr.shape[2] == 49, pr.shape

                    step_metrics["pred_max"] = list(pr.max(2))
                    step_metrics["spim_max"] = list(sp.max(2))
                    # step_metrics["pred"] = list(pr)
                    # step_metrics["spim"] = list(sp)
                    step_metrics["pred-vs-spim"] = list(torch.cat([pred, spim], dim=1))

                    # color version does not work somehow...
                    # zeros = torch.zeros_like(pred)
                    # pred = torch.cat([zeros, pred, pred], dim=1)
                    # spim = torch.cat([spim, zeros, spim], dim=1)
                    # step_metrics["pred-vs-spim"] = list(pred + spim)

                if self.run_logger is not None:
                    self.run_logger(
                        epoch=epoch, epoch_len=epoch_len, iteration=it, batch_len=batch["batch_len"], **step_metrics
                    )

                if self.save_spim_to_disk:
                    save_tensor_batch(self.save_spim_to_disk, batch[self.tgt_name])

            if self.save_pred_to_disk:
                save_tensor_batch(self.save_pred_to_disk, batch["pred"])

            sample_idx += batch["batch_len"]
            yield EvalYield(batch=batch, step_metrics=step_metrics)

        summary_metrics = self.metrics.compute()
        if self.save_pred_to_disk or self.save_spim_to_disk:
            summary_metrics["result_paths"] = pandas.DataFrame.from_dict(tab_data_per_step)

        self.run_logger.log_summary(**summary_metrics)
        self.metrics.reset()
        yield EvalYield(summary_metrics=summary_metrics)


class ValidationRun(EvalRun):
    def __init__(self, score_metric: str, minimize: bool, **super_kwargs):
        super().__init__(**super_kwargs)
        self.score_metric = score_metric
        self.minimize = minimize

    @no_grad()
    def get_validation_score(self) -> float:
        summary = None
        for y in self:
            summary = y.summary_metrics

        score = summary[self.score_metric]
        if self.minimize:
            score *= -1

        return score
