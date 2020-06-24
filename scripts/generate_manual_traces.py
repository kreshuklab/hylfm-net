import shutil
from pathlib import Path
from typing import Optional
from ruamel.yaml import YAML

import matplotlib.pyplot as plt
import numpy

from lnet.datasets.base import TensorInfo, TiffDataset
from lnet.utils.cache import cached_to_disk

yaml = YAML(typ="safe")


def get_data_to_trace(name: str, root: Path, tmin=0, tmax=None):

    ds = TiffDataset(info=TensorInfo(name=name, root=root, location=f"{name}/*.tif"))
    if tmax is None:
        tmax = len(ds)

    return numpy.stack([ds[t][name].squeeze() for t in range(tmin, tmax)])


class Tracker:
    def __init__(
        self, name: str, root: Path, save_traces_to: Path, tmin: int = 0, tmax: Optional[int] = None, radius: int = 3
    ):
        save_traces_to.mkdir(parents=True)
        self.save_traces_to = save_traces_to
        self.radius = radius
        self.tmin = tmin
        self.tmax = tmax
        self.video = get_data_to_trace(name, root=root, tmin=tmin, tmax=tmax)
        print("video", self.video.shape)
        self.T = self.video.shape[0]
        self._t = -1
        self._active_trace = -1
        self.traces = []
        self.trace_markers = []
        self.fig, self.ax = plt.subplots()
        cid = self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.fig.canvas.mpl_connect("scroll_event", self.on_scroll)
        self.im = self.ax.imshow(self.video[self.t])
        self.add_new_trace()
        plt.show()
        print(self.traces)
        for i, trace in enumerate(self.traces):
            yaml.dump([[x, y, radius] for x, y in trace.tolist()], save_traces_to / f"manual_trace_{i}.yml")

    @property
    def t(self):
        return self._t

    @t.setter
    def t(self, new_t):
        new_t = max(0, min(self.T - 1, new_t))
        if self._t != new_t:
            self.im.set_data(self.video[new_t])
            for i, ((x, y), peak_markers) in enumerate(
                zip([trace[new_t] for trace in self.traces], self.trace_markers)
            ):
                if peak_markers[self._t] is not None:
                    peak_markers[self._t].set_visible(False)

                if y != -1:
                    color = "red" if i == self.active_trace else "grey"
                    if peak_markers[new_t] is None:
                        peak_markers[new_t] = plt.Circle((y, x), self.radius, color=color, linewidth=1, fill=False)
                        self.ax.add_patch(peak_markers[new_t])
                    else:
                        peak_markers[new_t].set_visible(True)
                        peak_markers[new_t].set_color(color)
                        peak_markers[new_t].set_center((y, x))
                    # self.ax.text(x + 2 * int(self.radius + 0.5), y, str(i))

            self._t = new_t
            plt.draw()

        self.ax.set_title(f"time: {self.t}")

    @property
    def active_trace(self):
        return self._active_trace

    @active_trace.setter
    def active_trace(self, new_active_trace):
        new_active_trace = max(0, min(len(self.traces) - 1, new_active_trace))
        if new_active_trace != self._active_trace:
            self._active_trace = new_active_trace

    def on_click(self, event):
        if self.fig.canvas.manager.toolbar.mode:
            return

        print(
            f"{'double' if event.dblclick else 'single'} click: button={event.button}, x={event.x}, y={event.y}, xdata={event.xdata}, ydata={event.ydata}"
        )
        self.traces[self.active_trace][self.t] = [round(event.ydata), round(event.xdata)]
        self.t += 1

    def on_key(self, event):
        if event.key == "left":
            self.t -= 1
        elif event.key == "right":
            self.t += 1
        elif event.key == "up":
            self.active_trace += 1
        elif event.key == "down":
            self.active_trace -= 1
        elif event.key == "a":
            missing = numpy.where(self.traces[self.active_trace][:, 1] == -1)[0].tolist()
            if missing:
                print(f"{len(missing)} coordinates missing")
                self.t = missing[0] - 1
            else:
                print("all time points have a coordinate")
        elif event.key == "n":
            if -1 in self.traces[self.active_trace]:
                print("current trace not finished!")
            else:
                self.add_new_trace()

    def on_scroll(self, event):
        self.t += int(event.step)

    def add_new_trace(self):
        self.traces.append(numpy.full((self.T, 2), -1, dtype=numpy.int))
        self.trace_markers.append([None] * self.T)
        self.active_trace = len(self.traces) - 1
        self.t = 0


if __name__ == "__main__":
    name = "ls_slice"
    tag = "09_3__2020-03-09_06.43.40__SinglePlane_-330"
    save_traces_to = Path(f"C:/Users/fbeut/Desktop/lnet_stuff/manual_traces/{tag}/manual_on_{name}")
    Tracker(
        name=name,
        root=Path(f"C:/Users/fbeut/Desktop/lnet_stuff/manual_traces/{tag}"),
        save_traces_to=save_traces_to,
        # tmax=20,
    )
