"""AME with an actor-only long/short I/O history (LSIO) encoder.

Li et al., arXiv:2401.16889, Sec. V: temporal convolutions encode long
history while recent I/O samples bypass the encoder. This adapts that
architecture to G1/AME, retaining AME's terrain encoder and action decoder.
History is supplied by the environment, never advanced by the network.
"""

import torch
from torch import nn

from .actor_critic_encoder import ActorCriticEncoder


class ActorCriticEncoderLSIO(ActorCriticEncoder):
    def __init__(
        self,
        obs,
        obs_groups,
        num_actions,
        *,
        long_history_length=66,
        short_history_length=4,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        **kwargs,
    ):
        if actor_obs_normalization or critic_obs_normalization:
            raise ValueError("LSIO currently requires observation normalization to be disabled.")
        if not isinstance(long_history_length, int) or long_history_length < 15:
            raise ValueError("long_history_length must be an integer >= 15 for the two valid convolutions.")
        if not isinstance(short_history_length, int) or not 1 <= short_history_length <= long_history_length:
            raise ValueError("short_history_length must be between 1 and long_history_length.")
        history_groups = obs_groups.get("history", [])
        if len(history_groups) != 1 or history_groups[0] not in obs:
            raise ValueError("LSIO requires obs_groups['history'] to name one proprio_history observation group.")

        self.history_group = history_groups[0]
        self.long_history_length = long_history_length
        self.short_history_length = short_history_length
        # omega (3), gravity (3), joint position/velocity and previous action.
        self.history_frame_dim = 6 + 3 * num_actions
        self._validate_history(obs[self.history_group])
        conv_length = (long_history_length - 6) // 3 + 1
        conv_length = (conv_length - 4) // 2 + 1
        self.long_latent_dim = 16 * conv_length
        self.command_dim = 3

        super().__init__(
            obs, obs_groups, num_actions,
            actor_obs_normalization=False, critic_obs_normalization=False, **kwargs,
        )
        self.history_encoder = nn.Sequential(
            nn.Conv1d(self.history_frame_dim, 32, kernel_size=6, stride=3),
            nn.ReLU(),
            nn.Conv1d(32, 16, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten(start_dim=1),
        )

    def _actor_proprio_size(self, observation_dim):
        if observation_dim != self.command_dim:
            raise ValueError("LSIO policy observations must contain only 3 velocity commands followed by the map.")
        return self.long_latent_dim + self.short_history_length * self.history_frame_dim + self.command_dim

    def _validate_history(self, history):
        expected = (self.long_history_length, self.history_frame_dim)
        if history.ndim != 3 or tuple(history.shape[1:]) != expected:
            raise ValueError(f"LSIO history must have shape [batch, {expected[0]}, {expected[1]}], got {history.shape}.")

    def get_actor_obs(self, obs):
        current = super().get_actor_obs(obs)
        history = obs[self.history_group]
        self._validate_history(history)
        long_features = self.history_encoder(history.transpose(1, 2))
        short_features = history[:, -self.short_history_length:, :].flatten(start_dim=1)
        # Preserve the map-at-tail convention used by the shared terrain encoder.
        return torch.cat((long_features, short_features, current), dim=-1)

    def get_extra_state(self):
        # Tensor shapes alone cannot distinguish every history length (e.g. 66/67).
        return {
            "architecture": "ame_lsio_v1",
            "long_history_length": self.long_history_length,
            "short_history_length": self.short_history_length,
            "history_frame_dim": self.history_frame_dim,
            "map_scan_dim": tuple(self.map_scan_dim),
            "cnn_downsample": self.cnn_downsample,
            "attach_global": self.attach_global,
            "mha_dim": self.mha_dim,
            "num_heads": self.num_heads,
        }

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise RuntimeError("LSIO checkpoint architecture/history configuration does not match this task.")

    def load_state_dict(self, state_dict, strict=True):
        if "_extra_state" not in state_dict:
            raise RuntimeError("LSIO requires an LSIO checkpoint; legacy AME checkpoints cannot be loaded into this task.")
        self.set_extra_state(state_dict["_extra_state"])
        return super().load_state_dict(state_dict, strict=strict)
