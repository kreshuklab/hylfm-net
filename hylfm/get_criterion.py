import logging
from typing import TYPE_CHECKING

import hylfm.criteria
from hylfm.hylfm_types import CriterionChoice, TransformsPipeline
from hylfm.utils.general import Period

if TYPE_CHECKING:
    from hylfm.checkpoint import TrainRunConfig


logger = logging.getLogger(__name__)


def get_criterion(config: "TrainRunConfig", transforms_pipeline: TransformsPipeline):
    if config.criterion == CriterionChoice.L1:
        crit_kwargs = dict()
    elif config.criterion == CriterionChoice.MS_SSIM:
        crit_kwargs = dict(
            channel=1,
            data_range=config.data_range,
            size_average=True,
            spatial_dims=transforms_pipeline.spatial_dims,
            win_size=config.win_size,
            win_sigma=config.win_sigma,
        )
    elif config.criterion == CriterionChoice.MSE:
        crit_kwargs = dict()
    elif config.criterion == CriterionChoice.SmoothL1:
        crit_kwargs = dict(beta=config.crit_beta)
    elif config.criterion == CriterionChoice.SmoothL1_MS_SSIM:
        crit_kwargs = dict(
            beta=config.crit_beta,
            ms_ssim_weight=config.crit_ms_ssim_weight,
            channel=1,
            data_range=config.data_range,
            size_average=True,
            spatial_dims=transforms_pipeline.spatial_dims,
            win_size=config.win_size,
            win_sigma=config.win_sigma,
        )
    elif config.criterion == CriterionChoice.WeightedSmoothL1:
        crit_kwargs = dict(
            threshold=config.crit_threshold,
            weight=config.crit_weight,
            apply_weight_above_threshold=config.crit_apply_weight_above_threshold,
            beta=config.crit_beta,
            decay_weight_by=config.crit_decay_weight_by,
            decay_weight_every=Period(config.crit_decay_weight_every_value, config.crit_decay_weight_every_unit),
            decay_weight_limit=config.crit_decay_weight_limit,
        )
    elif config.criterion == CriterionChoice.WeightedSmoothL1_MS_SSIM:
        crit_kwargs = dict(
            threshold=config.crit_threshold,
            weight=config.crit_weight,
            apply_weight_above_threshold=config.crit_apply_weight_above_threshold,
            beta=config.crit_beta,
            decay_weight_by=config.crit_decay_weight_by,
            decay_weight_every=Period(config.crit_decay_weight_every_value, config.crit_decay_weight_every_unit),
            decay_weight_limit=config.crit_decay_weight_limit,
            ms_ssim_weight=config.crit_ms_ssim_weight,
            channel=1,
            data_range=config.data_range,
            size_average=True,
            spatial_dims=transforms_pipeline.spatial_dims,
            win_size=config.win_size,
            win_sigma=config.win_sigma,
        )
    else:
        raise NotImplementedError(config.criterion)

    crit_class = getattr(hylfm.criteria, config.criterion)
    try:
        crit = crit_class(**crit_kwargs)
    except Exception:
        logger.error("Failed to init %s with %s", crit_class, crit_kwargs)
        raise

    return crit
