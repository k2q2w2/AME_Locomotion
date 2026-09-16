"""Observation layout shared by the LSIO training and Play tasks."""

from copy import deepcopy

from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg
from isaaclab.utils import configclass


@configclass
class LSIOPolicyCfg(ObservationGroupCfg):
    # Field order is part of the network interface: commands, then XYZ map.
    velocity_commands: ObservationTermCfg = None
    height_scan: ObservationTermCfg = None


@configclass
class LSIOHistoryCfg(ObservationGroupCfg):
    # Each frame pairs the current state with the previously applied action.
    base_ang_vel: ObservationTermCfg = None
    projected_gravity: ObservationTermCfg = None
    joint_pos: ObservationTermCfg = None
    joint_vel: ObservationTermCfg = None
    actions: ObservationTermCfg = None
    history_length: int = 66
    flatten_history_dim: bool = False
    concatenate_terms: bool = True
    concatenate_dim: int = -1


@configclass
class LSIOObservationsCfg:
    policy: LSIOPolicyCfg = LSIOPolicyCfg()
    proprio_history: LSIOHistoryCfg = LSIOHistoryCfg()
    critic: ObservationGroupCfg = None


def make_lsio_observations(observations):
    """Split the already configured AME observations, preserving stage/play noise.

    Isaac Lab applies noise and scaling before appending to native term histories.
    Its buffers advance only on observation computation with update_history=True;
    reset fills just the reset environments with their first new frame.
    """
    source = observations.policy
    result = LSIOObservationsCfg()
    for name in ("velocity_commands", "height_scan"):
        setattr(result.policy, name, deepcopy(getattr(source, name)))
    for name in ("base_ang_vel", "projected_gravity", "joint_pos", "joint_vel", "actions"):
        setattr(result.proprio_history, name, deepcopy(getattr(source, name)))
    result.policy.enable_corruption = source.enable_corruption
    result.proprio_history.enable_corruption = source.enable_corruption
    result.critic = deepcopy(observations.critic)
    return result
