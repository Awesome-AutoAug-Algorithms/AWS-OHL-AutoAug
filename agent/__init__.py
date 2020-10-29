from .base import BasicAgent
from .ppo import PPOAgent as ppo
from .reinforce import Reinforce as REINFORCE


def agent_entry(dist, lg, agent_cfg) -> BasicAgent:
    return globals()[agent_cfg.type](
        dist=dist, lg=lg,
        **agent_cfg.kwargs if agent_cfg.kwargs is not None else {}
    )
