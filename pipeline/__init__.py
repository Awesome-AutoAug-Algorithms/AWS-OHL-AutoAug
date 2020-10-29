from .aws import AWSPipeline as aws
from .base import BasicPipeline
from .ohl import OHLPipeline as ohl


def pipeline_entry(dist, pp_cfg) -> BasicPipeline:
    return globals()[pp_cfg.type](
        dist=dist,
        common_kwargs=pp_cfg.common_kwargs,
        **pp_cfg.special_kwargs
    )
