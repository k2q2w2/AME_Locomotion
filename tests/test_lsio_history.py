"""Native Isaac Lab observation-manager CPU tests with synthetic sensor values.

No SimulationApp/physics is started. Only the unused omni.timeline import is
stubbed; configclass, manager configuration, ObservationManager and its buffers
are loaded from the installed Isaac Lab source unchanged.
"""

import ast
import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "source/ame_locomotion/ame_locomotion/tasks/manager_based/ame_locomotion"
NAMES = ("base_ang_vel", "projected_gravity", "joint_pos", "joint_vel", "actions")
WIDTHS = (3, 3, 29, 29, 29)


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def native():
    isaaclab = pytest.importorskip("isaaclab")
    from isaaclab.utils import configclass
    from isaaclab.utils.noise import UniformNoiseCfg

    package = ModuleType("_lsio_native_managers")
    package.__path__ = [str(Path(isaaclab.__file__).parent / "managers")]
    sys.modules[package.__name__] = package
    # Bypass managers/__init__.py, which eagerly loads simulation-only managers.
    omni = ModuleType("omni")
    omni.timeline = ModuleType("omni.timeline")
    with patch.dict(sys.modules, {"omni": omni, "omni.timeline": omni.timeline}):
        manager = importlib.import_module("_lsio_native_managers.observation_manager")
        terms = importlib.import_module("_lsio_native_managers.manager_term_cfg")
    package.ObservationGroupCfg = terms.ObservationGroupCfg
    package.ObservationTermCfg = terms.ObservationTermCfg
    with patch.dict(sys.modules, {"isaaclab.managers": package}):
        layouts = load_file("_lsio_layouts", TASK / "29dof/lsio_observations.py")

    # Evaluate the original observation definitions without loading robot/USD assets.
    source = ast.parse((TASK / "29dof/velocity_env_cfg_29dof.py").read_text())
    definition = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == "ObservationsCfg")
    mdp = SimpleNamespace(**{name: (lambda env: None) for name in (
        "base_ang_vel", "projected_gravity", "generated_commands", "joint_pos_rel",
        "joint_vel_rel", "last_action", "elevation_map", "base_lin_vel",
    )})
    namespace = {
        "configclass": configclass, "ObsGroup": terms.ObservationGroupCfg,
        "ObsTerm": terms.ObservationTermCfg, "Unoise": UniformNoiseCfg,
        "mdp": mdp, "SceneEntityCfg": terms.SceneEntityCfg,
    }
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(TASK / "29dof/velocity_env_cfg_29dof.py"), "exec"), namespace)
    return SimpleNamespace(manager=manager.ObservationManager, layouts=layouts, namespace=namespace, source=source)


class ConfigTree(SimpleNamespace):
    """Config-only stand-in for scene/reward fields touched by parent post-init."""

    def __getattr__(self, name):
        value = ConfigTree()
        setattr(self, name, value)
        return value


def configured_source(native, finetune, play=False):
    namespace = dict(native.namespace, FINETUNE=finetune)
    source = namespace["ObservationsCfg"]()
    # Run the complete original train post-init on config-only scene/reward trees.
    cls = next(node for node in native.source.body if isinstance(node, ast.ClassDef) and node.name == "G1RoughEnvCfg")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "__post_init__")
    exec(compile(ast.Module(body=[method], type_ignores=[]), "train_post_init", "exec"), namespace)
    cfg = ConfigTree(observations=source)
    namespace["__post_init__"](cfg)
    assert cfg.decimation * cfg.sim.dt == 0.02
    if play:
        # Execute the actual Play observation overrides, without constructing cameras/terrains.
        cls = next(node for node in native.source.body if isinstance(node, ast.ClassDef) and node.name == "G1RoughEnvCfg_PLAY")
        method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "__post_init__")
        overrides = [node for node in method.body if isinstance(node, ast.Assign)
                     and ast.unparse(node.targets[0]).startswith("self.observations.")]
        assert len(overrides) == 2
        exec(compile(ast.Module(body=overrides, type_ignores=[]), "play_observation_overrides", "exec"), dict(namespace, self=cfg))
    return source


@pytest.mark.parametrize("finetune,play", [(False, False), (True, False), (False, True), (True, True)])
def test_stage_and_play_settings(native, finetune, play):
    source = configured_source(native, finetune, play)
    cfg = native.layouts.make_lsio_observations(source)
    history = cfg.proprio_history
    assert (history.history_length, history.flatten_history_dim, history.concatenate_dim) == (66, False, -1)
    assert history.enable_corruption == (not play)
    terms = [name for name, value in vars(history).items() if name in NAMES]
    assert terms == list(NAMES)
    for name in NAMES:
        assert getattr(history, name).scale == getattr(source.policy, name).scale
        assert getattr(history, name) is not getattr(source.policy, name)
        assert (getattr(history, name).noise is not None) == (finetune and name != "actions")
    assert cfg.policy.height_scan.params["noise"] == (finetune and not play)
    assert cfg.critic.to_dict() == source.critic.to_dict()
    assert cfg.critic is not source.critic
    assert not hasattr(history, "velocity_commands")
    assert not hasattr(history, "height_scan")
    assert not hasattr(history, "base_lin_vel")


def read_field(env, name):
    return env.values[name]


def make_manager(native, *, noisy=False):
    source = configured_source(native, finetune=noisy)
    cfg = native.layouts.make_lsio_observations(source)
    values = {name: torch.zeros(2, width) for name, width in zip(NAMES, WIDTHS)}
    env = SimpleNamespace(num_envs=2, device="cpu", values=values, sim=SimpleNamespace(is_playing=lambda: True))
    for name in NAMES:
        term = getattr(cfg.proprio_history, name)
        term.func = read_field
        term.params = {"name": name}
    manager = native.manager({"proprio_history": cfg.proprio_history}, env)
    assert manager.group_obs_dim["proprio_history"] == (66, 93)
    return manager, env


def test_native_order_pairing_reset_and_reads(native):
    manager, env = make_manager(native)
    # Read before first control step must not initialize the persistent buffers.
    manager.compute()
    buffers = manager._group_obs_term_history_buffer["proprio_history"]
    assert all(buffer._buffer is None for buffer in buffers.values())
    for step in range(70):
        for name in NAMES:
            # s_t paired with a_(t-1); reset actions are zero in the real env.
            env.values[name].fill_(step if name != "actions" else step - 1)
        obs = manager.compute(update_history=True)["proprio_history"]
    torch.testing.assert_close(obs[0, :, 0], torch.arange(4, 70).float() * 0.2)
    torch.testing.assert_close(obs[0, :, 3], torch.arange(4, 70).float())
    torch.testing.assert_close(obs[0, :, 35], torch.arange(4, 70).float() * 0.05)
    torch.testing.assert_close(obs[0, :, 64], torch.arange(3, 69).float())
    torch.testing.assert_close(obs[0, -4:, 64], torch.arange(65, 69).float())
    snapshot = obs.clone()
    for _ in range(3):
        torch.testing.assert_close(manager.compute()["proprio_history"], snapshot, rtol=0, atol=0)
    manager.reset(env_ids=torch.tensor([0]))
    for name in NAMES:
        env.values[name][0] = 0 if name == "actions" else -10
        env.values[name][1] = 70 if name != "actions" else 69
    obs = manager.compute(update_history=True)["proprio_history"]
    torch.testing.assert_close(obs[0], obs[0, -1:].expand(66, -1), rtol=0, atol=0)
    assert (obs[0, :, 64:] == 0).all()
    torch.testing.assert_close(obs[1, :-1], snapshot[1, 1:], rtol=0, atol=0)


def test_native_noise_is_stored_once_before_scaling(native):
    torch.manual_seed(4)
    manager, env = make_manager(native, noisy=True)
    first = manager.compute(update_history=True)["proprio_history"]
    assert first[:, :, :64].abs().sum() > 0
    assert first[:, :, :3].abs().max() <= 0.2 * 0.2
    assert first[:, :, 35:64].abs().max() <= 2.0 * 0.05
    assert (first[:, :, 64:] == 0).all()
    torch.testing.assert_close(first, first[:, -1:].expand(-1, 66, -1), rtol=0, atol=0)
    second = manager.compute(update_history=True)["proprio_history"]
    torch.testing.assert_close(second[:, :-1], first[:, 1:], rtol=0, atol=0)
    assert not torch.equal(second[:, -1], first[:, -1])
    for _ in range(3):
        torch.testing.assert_close(manager.compute()["proprio_history"], second, rtol=0, atol=0)


def test_all_observation_groups_and_runner_configuration(native):
    cfg = native.layouts.make_lsio_observations(configured_source(native, finetune=False))
    widths = dict(zip(NAMES, WIDTHS), velocity_commands=3, base_lin_vel=3, height_scan=2079)
    env = SimpleNamespace(num_envs=2, device="cpu", sim=SimpleNamespace(is_playing=lambda: True),
                          values={name: torch.zeros(2, width) for name, width in widths.items()})
    for group in vars(cfg).values():
        for name, term in vars(group).items():
            if name in widths:
                term.func, term.params = read_field, {"name": name}
    manager = native.manager(cfg, env)
    obs = manager.compute(update_history=True)
    assert {key: tuple(value.shape) for key, value in obs.items()} == {
        "policy": (2, 2082), "proprio_history": (2, 66, 93), "critic": (2, 2178),
    }
    # Load the real runner config without eagerly importing simulation wrappers.
    isaaclab_rl = pytest.importorskip("isaaclab_rl")
    package = ModuleType("_lsio_rl_cfg")
    package.__path__ = [str(Path(isaaclab_rl.__file__).parent / "rsl_rl")]
    sys.modules[package.__name__] = package
    rl_cfg = importlib.import_module("_lsio_rl_cfg.rl_cfg")
    with patch.dict(sys.modules, {"isaaclab_rl.rsl_rl": rl_cfg}):
        agent = load_file("_lsio_agent_cfg", TASK / "agents/ame_rsl_rl_ppo_cfg.py")
    old = agent.G1AMEPPORunnerCfg()
    new = agent.G1AMELSIOPPORunnerCfg()
    assert new.experiment_name == "g1_ame_lsio"
    assert new.policy.class_name == "ActorCriticEncoderLSIO"
    assert new.policy.long_history_length == 66
    assert new.policy.short_history_length == 4
    assert new.policy.attach_global is False
    assert new.obs_groups == {"policy": ["policy"], "critic": ["critic"], "history": ["proprio_history"]}
    assert new.algorithm.to_dict() == old.algorithm.to_dict()
    assert new.num_steps_per_env == old.num_steps_per_env
    assert new.policy.actor_obs_normalization is False
    assert new.policy.critic_obs_normalization is False
    from rsl_rl.runners import OnPolicyRunner
    from tensordict import TensorDict

    env.num_actions = 29
    env.get_observations = lambda: TensorDict(obs, batch_size=[2])
    runner = OnPolicyRunner(env, new.to_dict(), device="cpu")
    actions, weights = runner.get_inference_policy()(env.get_observations())
    assert actions.shape == (2, 29)
    assert weights.shape == (2, 1, 187)
