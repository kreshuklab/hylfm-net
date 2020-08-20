import torch
import torch.nn as nn
import torch.nn.functional


class C2Z(nn.Module):
    def __init__(self, z_out: int):
        super().__init__()
        self.z_out = z_out

    def get_c_out(self, c_in: int):
        c_out = c_in // self.z_out
        assert c_in == c_out * self.z_out, (c_in, c_out, self.z_out)
        return c_out

    def forward(self, input: torch.Tensor):
        c_out = self.get_c_out(input.shape[1])
        return input.view(input.shape[0], c_out, self.z_out, *input.shape[2:])


class Interpolate(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.interp = nn.functional.interpolate
        self._kwargs = kwargs

    def forward(self, input):
        return self.interp(input, **self._kwargs)


class RepeatInterleave(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self._kwargs = kwargs

    def forward(self, x):
        for dim in self._kwargs["dims"]:
            x = torch.repeat_interleave(x, 2, dim=dim)

        return x


class Crop(nn.Module):
    def __init__(self, *slices: slice):
        super().__init__()
        self.slices = slices

    def extra_repr(self):
        return str(self.slices)

    def forward(self, input):
        return input[self.slices]
