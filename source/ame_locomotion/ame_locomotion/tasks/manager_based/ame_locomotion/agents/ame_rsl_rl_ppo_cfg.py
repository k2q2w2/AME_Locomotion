# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

@configclass
class G1AMEPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 100
    experiment_name = "g1_ame"
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCriticEncoder",
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.008,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class LSIOActorCriticCfg(RslRlPpoActorCriticCfg):
    class_name: str = "ActorCriticEncoderLSIO"
    long_history_length: int = 66
    short_history_length: int = 4
    attach_global: bool = False


@configclass
class G1AMELSIOPPORunnerCfg(G1AMEPPORunnerCfg):
    experiment_name = "g1_ame_lsio"
    obs_groups = {"policy": ["policy"], "critic": ["critic"], "history": ["proprio_history"]}
    policy = LSIOActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )


@configclass
class GLADActorCriticCfg(LSIOActorCriticCfg):
    class_name: str = "ActorCriticEncoderGLAD"
    attach_global: bool = True
    top_k: int = 32
    gumbel_temperature: float = 1.0


@configclass
class G1AMEGLADPPORunnerCfg(G1AMELSIOPPORunnerCfg):
    experiment_name = "g1_ame_glad"
    policy = GLADActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
