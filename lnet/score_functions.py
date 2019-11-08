from lnet.engine import TunedEngine

from lnet.metrics import (
    LOSS_NAME,
    AUX_LOSS_NAME,
    NRMSE_NAME,
    PSNR_NAME,
    SSIM_NAME,
    MSSSIM_NAME,
    BEAD_PRECISION_RECALL,
)


def rename(newname):
    def decorator(f):
        f.__name__ = newname
        return f

    return decorator


@rename("ms-ssim*100+f1")
def msssim100f1(engine):
    score = round(engine.state.metrics[MSSSIM_NAME] * 100)
    p, r = engine.state.metrics.get(BEAD_PRECISION_RECALL, (0, 0))
    if p and r:
        score += 2 * p * r / (p + r)

    return score


def generic_metric_score(metric_name: str):
    def score_fn(engine: TunedEngine):
        return engine.state.metrics[metric_name]

    return score_fn

known_score_functions = {
    msssim100f1.__name__: msssim100f1,
    **{
        metric_name: generic_metric_score(metric_name)
        for metric_name in [
            LOSS_NAME,
            AUX_LOSS_NAME,
            NRMSE_NAME,
            PSNR_NAME,
            SSIM_NAME,
            MSSSIM_NAME,
            BEAD_PRECISION_RECALL,
        ]
    },
}

if __name__ == "__main__":
    import inspect
    from lnet.engine import TunedEngine

    print(known_score_functions)
    sfn = known_score_functions[MSSSIM_NAME]
    print(sfn)
    print(inspect.signature(sfn))
