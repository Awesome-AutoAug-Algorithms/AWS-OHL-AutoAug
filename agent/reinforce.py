from .base import BasicAgent


class Reinforce(BasicAgent):
    
    def __init__(
            self, dist, lg, op_cfg,
    ):
        super(Reinforce, self).__init__(dist=dist, lg=lg, op_cfg=op_cfg)
    
    def _ppo_loss(self):
        pass
    
    def _step_process(self, advance, **kwargs):
        pass
