"""Actor-only LSIO observations; rewards, actions and simulation remain AME's."""

from isaaclab.utils import configclass

from .lsio_observations import make_lsio_observations
from .velocity_env_cfg_29dof import G1RoughEnvCfg, G1RoughEnvCfg_PLAY


@configclass
class G1LSIOEnvCfg(G1RoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations = make_lsio_observations(self.observations)


@configclass
class G1LSIOEnvCfg_PLAY(G1RoughEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        self.observations = make_lsio_observations(self.observations)
