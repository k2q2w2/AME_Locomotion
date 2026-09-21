"""GLAD terrain encoding with LSIO and optional clean Actor features for values.

Fu et al., arXiv:2606.00637v3, Sec. III-D and IV-A: attention pooling
summarizes all terrain tokens; saliency selection retains sparse keys/values
for state-conditioned MHA. Observations and PPO remain those of AME/LSIO.
"""

import math

import torch
from torch import nn

from .actor_critic_encoder_lsio import ActorCriticEncoderLSIO


class ActorCriticEncoderGLAD(ActorCriticEncoderLSIO):
    critic_feature_sources = ("actor", "critic", "actor_clean")

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions,
        *,
        top_k=32,
        gumbel_temperature=1.0,
        attach_global=True,
        **kwargs,
    ):
        if not attach_global:
            raise ValueError("GLAD requires attach_global=True for its global attention branch.")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError("GLAD top_k must be a positive integer.")
        if not math.isfinite(gumbel_temperature) or gumbel_temperature <= 0:
            raise ValueError("GLAD gumbel_temperature must be finite and positive.")
        self.top_k = top_k
        self.gumbel_temperature = float(gumbel_temperature)
        super().__init__(obs, obs_groups, num_actions, attach_global=True, **kwargs)
        if self.critic_feature_source == "actor_clean":
            groups = obs_groups.get("clean_history", [])
            if len(groups) != 1 or groups[0] not in obs:
                raise ValueError("actor_clean requires obs_groups['clean_history'] to name a clean history group.")
            self.clean_history_group = groups[0]
            if self.clean_history_group == self.history_group:
                raise ValueError("actor_clean requires a separate clean history observation group.")
            self._validate_history(obs[self.clean_history_group])
            # Current Critic layout: v, omega, gravity, command, q, dq, previous action.
            if self.critic_proprio_dim != 12 + 3 * num_actions:
                raise ValueError("actor_clean requires the standard Critic proprioception layout.")

    def get_clean_actor_obs(self, obs):
        """Use Actor LSIO parameters with clean history, commands and XYZ map.

        Called inside the value path's no_grad context. Do not call act() or
        update_distribution(): bootstrap must preserve the current policy.
        """
        critic_obs = self.get_critic_obs(obs)
        long_features, short_features = self._encode_history(obs[self.clean_history_group])
        commands = critic_obs[:, 9:12]
        terrain = critic_obs[:, self.critic_proprio_dim:]
        return torch.cat((long_features, short_features, commands, terrain), dim=-1)

    def _build_terrain_encoder(self, actor_proprio_dim, critic_proprio_dim, attach_global):
        if self.coord_dim != 3:
            raise ValueError("GLAD expects an XYZ map with three coordinate channels.")
        self.num_terrain_tokens = (
            ((self.L + 1) // 2) * ((self.W + 1) // 2) if self.cnn_downsample else self.L * self.W
        )
        if self.top_k > self.num_terrain_tokens:
            raise ValueError(f"GLAD top_k must not exceed {self.num_terrain_tokens} terrain tokens.")
        super()._build_terrain_encoder(actor_proprio_dim, critic_proprio_dim, attach_global)
        # Replace AME2's pointwise MLP/max-pool branch. No unused MLP remains
        # registered. CNN, these scorers, query fusion and MHA are shared.
        self.global_encoder = nn.Linear(self.mha_dim, 1)
        self.saliency_score = nn.Linear(self.mha_dim, 1)

    def _select_local_features(self, local_features):
        scores = self.saliency_score(local_features).squeeze(-1)
        if self.training:
            eps = torch.finfo(scores.dtype).eps
            uniform = torch.rand_like(scores).clamp(min=eps, max=1.0 - eps)
            scores = scores - torch.log(-torch.log(uniform))
        indices = scores.topk(self.top_k, dim=-1).indices
        selected = local_features.gather(1, indices.unsqueeze(-1).expand(-1, -1, self.mha_dim))
        if self.training:
            # Hard Top-K forward, softmax surrogate backward (tau=1 by default).
            # The paper specifies ST weighting but not its exact implementation;
            # this gate leaves the selected feature values unchanged forward.
            probabilities = torch.softmax(scores / self.gumbel_temperature, dim=-1)
            selected_probabilities = probabilities.gather(1, indices)
            gate = 1.0 + (selected_probabilities - selected_probabilities.detach())
            selected = selected * gate.unsqueeze(-1)
        return selected, indices

    def _encode_terrain(self, obs, *, role="actor", update_buffers=True):
        map_size = self.L * self.W * self.coord_dim
        map_scan = obs[:, -map_size:].reshape(-1, self.W, self.L, self.coord_dim)
        local_features = self._encode_map(
            map_scan.permute(0, 3, 1, 2), update_buffers=update_buffers,
        ).flatten(2).transpose(1, 2)
        proprio_obs = obs[:, :-map_size]
        if role == "actor":
            proprio_embedding = self.actor_proprio_embedding(proprio_obs)
        elif role == "critic":
            proprio_embedding = self.critic_proprio_embedding(proprio_obs)
        else:
            raise ValueError(f"Unknown terrain encoder role: {role}")

        global_weights = torch.softmax(self.global_encoder(local_features), dim=1)
        global_features = (global_weights * local_features).sum(dim=1)
        query = self.query_projector(torch.cat((global_features, proprio_embedding), dim=-1)).unsqueeze(1)
        selected_features, indices = self._select_local_features(local_features)
        attended, sparse_weights = self.mha(query=query, key=selected_features, value=selected_features)

        # Preserve the full spatial grid expected by the existing Play visualizer.
        attention_weights = sparse_weights.new_zeros(obs.shape[0], 1, local_features.shape[1])
        attention_weights = attention_weights.scatter(2, indices.unsqueeze(1), sparse_weights)
        encoded = torch.cat((global_features, attended.squeeze(1), proprio_obs), dim=-1)
        return encoded, attention_weights

    def get_extra_state(self):
        state = super().get_extra_state()
        state.update(
            architecture=(
                "glad_lsio_clean_actor_v1" if self.critic_feature_source == "actor_clean" else "glad_lsio_v1"
            ),
            top_k=self.top_k,
            gumbel_temperature=self.gumbel_temperature,
        )
        return state

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise RuntimeError("GLAD checkpoint architecture/history/selection configuration does not match this task.")

    def load_state_dict(self, state_dict, strict=True):
        self.set_extra_state(state_dict.get("_extra_state"))
        return super().load_state_dict(state_dict, strict=strict)
