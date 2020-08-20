from concurrent.futures import Future
from enum import Enum


class PeriodUnit(Enum):
    epoch = "epoch"
    iteration = "iteration"


class Period:
    def __init__(self, value: int, unit: str):
        self.value = value
        self.unit = PeriodUnit(unit)


class DummyPool:
    def __init__(self, *args, **kwargs):
        pass

    def submit(self, fn, *args, **kwargs):
        fut = Future()
        fut.set_result(fn(*args, **kwargs))
        return fut
