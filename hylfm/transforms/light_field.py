from .base import Transform
from ..hylfm_types import Array


class ChannelFromLightField(Transform):
    def __init__(self, nnum: int, **super_kwargs):
        super().__init__(**super_kwargs)
        self.nnum = nnum

    def apply_to_sample(self, tensor: Array):
        assert len(tensor.shape) == 3, tensor.shape
        c, x, y = tensor.shape
        assert c == 1
        assert x % self.nnum == 0, (x, self.nnum)
        assert y % self.nnum == 0, (y, self.nnum)
        return (
            tensor.reshape(x // self.nnum, self.nnum, y // self.nnum, self.nnum)
            .transpose(1, 3, 0, 2)
            .reshape(self.nnum ** 2, x // self.nnum, y // self.nnum)
        )


class LightFieldFromChannel(Transform):
    def __init__(self, nnum: int, **super_kwargs):
        super().__init__(**super_kwargs)
        self.nnum = nnum

    def apply_to_sample(self, tensor: Array):
        assert len(tensor.shape) == 3
        c, x, y = tensor.shape
        assert c == self.nnum ** 2
        return tensor.reshape(self.nnum, self.nnum, x, y).transpose(2, 0, 3, 1).reshape(1, self.nnum * x, self.nnum * y)
