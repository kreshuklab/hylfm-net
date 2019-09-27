from lnet.utils.metrics import (
    LOSS_NAME,
    AUX_LOSS_NAME,
    NRMSE_NAME,
    PSNR_NAME,
    SSIM_NAME,
    MSSSIM_NAME,
    BEAD_PRECISION_RECALL,
    BEAD_PRECISION,
    BEAD_RECALL,
)


def rename(newname):
    def decorator(f):
        f.__name__ = newname
        return f

    return decorator


@rename("ms-ssim*100+f1")
def msssim100f1(engine):
    score = round(engine.state.metrics[MSSSIM_NAME] * 100)
    p = engine.state.metrics.get(BEAD_PRECISION, 0)
    r = engine.state.metrics.get(BEAD_RECALL, 0)
    if p and r:
        score += 2 * p * r / (p + r)

    return score


known_score_functions = {
    msssim100f1.__name__: msssim100f1,
    **{
        metric_name: lambda engine: engine.state.metrics[metric_name]
        for metric_name in [
            LOSS_NAME,
            AUX_LOSS_NAME,
            NRMSE_NAME,
            PSNR_NAME,
            SSIM_NAME,
            MSSSIM_NAME,
            BEAD_PRECISION_RECALL,
            BEAD_PRECISION,
            BEAD_RECALL,
        ]
    },
}
