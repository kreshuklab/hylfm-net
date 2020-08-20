from lnet.transformations.base import Transform


class ChannelFromLightField(Transform):
    def __init__(self, nnum: int, **super_kwargs):
        super().__init__(**super_kwargs)
        self.nnum = nnum

    def apply_to_sample(self, sample, *, tensor_name: str, tensor_idx: int, batch_idx: int, meta: dict):
        assert len(sample.shape) == 3
        c, x, y = sample.shape
        assert c == 1
        assert x % self.nnum == 0, (x, self.nnum)
        assert y % self.nnum == 0, (y, self.nnum)
        return (
            sample.reshape(x // self.nnum, self.nnum, y // self.nnum, self.nnum)
            .transpose(1, 3, 0, 2)
            .reshape(self.nnum ** 2, x // self.nnum, y // self.nnum)
        )


class LightFieldFromChannel(Transform):
    def __init__(self, nnum: int, **super_kwargs):
        super().__init__(**super_kwargs)
        self.nnum = nnum

    def apply_to_sample(self, sample, *, tensor_name: str, tensor_idx: int, batch_idx: int, meta: dict):
        assert len(sample.shape) == 3
        c, x, y = sample.shape
        assert c == self.nnum ** 2
        return sample.reshape(self.nnum, self.nnum, x, y).transpose(2, 0, 3, 1).reshape(1, self.nnum * x, self.nnum * y)
