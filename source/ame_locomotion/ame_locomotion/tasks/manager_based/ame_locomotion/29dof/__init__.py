import gymnasium as gym
from ame_locomotion.tasks.manager_based.ame_locomotion import agents

for play in (False, True):
    gym.register(
        id=f"AME-G1-29DOF-GLAD-CriticCleanStopGrad{'-Play' if play else ''}-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.lsio_env_cfg:G1LSIOEnvCfg{'_PLAY' if play else ''}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.ame_rsl_rl_ppo_cfg:G1AMEGLADCriticCleanStopGradPPORunnerCfg",
        },
    )

# Ablations reuse the original environments, rewards, and PPO configurations.
for variant, env_cfg, runner_cfg in (
    ("", "velocity_env_cfg_29dof:G1RoughEnvCfg", "G1AMECriticStopGradPPORunnerCfg"),
    ("-LSIO", "lsio_env_cfg:G1LSIOEnvCfg", "G1AMELSIOCriticStopGradPPORunnerCfg"),
    ("-GLAD", "lsio_env_cfg:G1LSIOEnvCfg", "G1AMEGLADCriticStopGradPPORunnerCfg"),
):
    for play in (False, True):
        gym.register(
            id=f"AME-G1-29DOF{variant}-CriticStopGrad{'-Play' if play else ''}-v0",
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"{__name__}.{env_cfg}{'_PLAY' if play else ''}",
                "rsl_rl_cfg_entry_point": f"{agents.__name__}.ame_rsl_rl_ppo_cfg:{runner_cfg}",
            },
        )

gym.register(
    id="AME-G1-29DOF-GLAD-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.lsio_env_cfg:G1LSIOEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ame_rsl_rl_ppo_cfg:G1AMEGLADPPORunnerCfg",
    },
)

gym.register(
    id="AME-G1-29DOF-GLAD-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.lsio_env_cfg:G1LSIOEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ame_rsl_rl_ppo_cfg:G1AMEGLADPPORunnerCfg",
    },
)

gym.register(
    id="AME-G1-29DOF-LSIO-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.lsio_env_cfg:G1LSIOEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ame_rsl_rl_ppo_cfg:G1AMELSIOPPORunnerCfg",
    },
)

gym.register(
    id="AME-G1-29DOF-LSIO-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.lsio_env_cfg:G1LSIOEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ame_rsl_rl_ppo_cfg:G1AMELSIOPPORunnerCfg",
    },
)

gym.register(
    id="AME-G1-29DOF-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_29dof:G1RoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ame_rsl_rl_ppo_cfg:G1AMEPPORunnerCfg",
        # "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
    },
)

gym.register(
    id="AME-G1-29DOF-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_29dof:G1RoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ame_rsl_rl_ppo_cfg:G1AMEPPORunnerCfg",
        # "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
    },
)
