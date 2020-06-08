from __future__ import annotations
from pathlib import Path
from typing import Union, Optional
import typing

from torch.utils.tensorboard import SummaryWriter

if typing.TYPE_CHECKING:
    from lnet.setup import Stage

from lnet.utils.tracer import trace


def trace_neurons(stage: Stage, tgt_path: Optional[Union[Path, str]] = None, tgt="ls_slice", compare_to: typing.Sequence[str] = ("pred", )):
    # if tgt_path is None:
    #     tgt = stage.log_path
    for ds_out_path in stage.log_path.glob("ds*-*"):
        if tgt_path is None:
            this_tgt_path = ds_out_path
            this_compare_to = set(compare_to)
        else:
            this_tgt_path = tgt_path
            this_compare_to = {ct: ds_out_path for ct in compare_to}

        peaks, peak_pos_figs, traces = trace(tgt_path=this_tgt_path, tgt=tgt, compare_to=this_compare_to)
        tbl = stage.log.loggers.get("TensorBoardLogger", None)
        if tbl is not None:
            tb_writer: SummaryWriter = tbl.backend.writer
            for name, fig in peak_pos_figs.items():
                tb_writer.add_figure(f"{stage.name}/{ds_out_path.name}-{name.replace(' ', '_')}", fig)



if __name__ == "__main__":
    class DummyStage:
        log_path = Path("/g/kreshuk/LF_computed/lnet/logs/brain1/z_out49/f2_only11_2/20-05-19_12-27-16/test/run000")
        class log:
            loggers = {}

    trace_neurons(stage=DummyStage())
