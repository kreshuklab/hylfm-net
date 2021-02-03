from hylfm import criteria
from hylfm.metrics.base import SimpleSingleValueMetric


class L1(SimpleSingleValueMetric, criteria.L1Loss):
    pass


class MSE(SimpleSingleValueMetric, criteria.MSELoss):
    pass


class SmoothL1(SimpleSingleValueMetric, criteria.SmoothL1Loss):
    pass


class SSIM(SimpleSingleValueMetric, criteria.SSIM):
    pass


class MS_SSIM(SimpleSingleValueMetric, criteria.MS_SSIM):
    pass


class WeightedL1(SimpleSingleValueMetric, criteria.WeightedL1Loss):
    pass


class WeightedSmoothL1(SimpleSingleValueMetric, criteria.WeightedSmoothL1Loss):
    pass


if __name__ == "__main__":
    metric = MS_SSIM(data_range=1, channel=1)

    print(metric)
    print(str(metric))
    print(repr(metric))
    # ipt = torch.randn((1, 1, 161, 161))
    # tgt = torch.randn((1, 1, 161, 161))
    #
    # print(ipt.min(), ipt.max())
    #
    # metric.update_with_batch(ipt, tgt)
    # print(metric.compute())
    # ipt = torch.randn((1, 161, 161))
    # tgt = torch.randn((1, 161, 161))
    #
    # metric.update_with_sample(ipt, tgt)
    # print(metric.compute())
