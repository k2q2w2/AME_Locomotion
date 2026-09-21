"""First-episode terrain traversal outcomes, independent of Isaac Sim imports."""
import torch

OUTCOMES = {0: 'running', 1: 'success', 2: 'illegal_contact', 3: 'out_of_bounds', 4: 'timeout'}


def traversal_outcome(position, contact, timed_out, supported, upright, *, goal_x=3.5, half_width=1.0):
    """Failure beats success on the same step; timeout alone is never success.

    Positions are relative to each terrain's central origin. A supported upright
    crossing must happen before the far tile boundary; surviving in place fails.
    """
    outside = (position[:, 1].abs() > half_width) | (position[:, 0] < -1.0) | (position[:, 0] > 3.9)
    reached = (position[:, 0] >= goal_x) & supported & upright
    result = torch.zeros_like(contact, dtype=torch.long)
    result[timed_out] = 4
    result[reached] = 1
    result[outside] = 3
    result[contact] = 2
    return result
