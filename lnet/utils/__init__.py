from concurrent.futures import Future


class DummyPool:
    def __init__(self, *args, **kwargs):
        pass

    def submit(self, fn, *args, **kwargs):
        fut = Future()
        fut.set_result(fn(*args, **kwargs))
        return fut
