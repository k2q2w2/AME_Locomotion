"""CPU coverage of shared GLAD encoding and the unchanged LSIO/PPO interface."""

import ast
import importlib
import sys
from copy import deepcopy
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from rsl_rl.modules import ActorCriticEncoderGLAD, ActorCriticEncoderLSIO
from rsl_rl.runners import OnPolicyRunner
from test_lsio import GROUPS, SyntheticEnv, observations, runner_config
from test_lsio_history import TASK, load_file


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.manual_seed(42)
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("role", ["actor", "critic"])
def test_both_losses_train_the_shared_encoder(role):
    obs = observations()
    model = ActorCriticEncoderGLAD(obs, GROUPS, 29)
    assert model.actor[0].in_features == 647
    assert model.critic[0].in_features == 227
    assert model.is_recurrent is False
    if role == "actor":
        model.act(obs)
        loss = model.action_mean.square().mean()
    else:
        loss = model.evaluate(obs).square().mean()
    loss.backward()
    # Both roles reach the very same parameter attributes, including selection.
    for layer in (model.map_cnn[0], model.global_encoder, model.saliency_score, model.query_projector):
        assert layer.weight.grad is not None
        assert torch.isfinite(layer.weight.grad).all()
        assert layer.weight.grad.abs().sum() > 0
    assert model.mha.in_proj_weight.grad.abs().sum() > 0
    active = model.actor_proprio_embedding if role == "actor" else model.critic_proprio_embedding
    inactive = model.critic_proprio_embedding if role == "actor" else model.actor_proprio_embedding
    assert active.weight.grad.abs().sum() > 0
    assert inactive.weight.grad is None
    if role == "actor":
        assert model.history_encoder[0].weight.grad.abs().sum() > 0
    else:
        assert model.history_encoder[0].weight.grad is None


@pytest.mark.parametrize("top_k,downsample", [(1, True), (32, True), (187, True), (32, False)])
def test_hard_selection_pooling_and_spatial_attention(top_k, downsample):
    obs = observations()
    model = ActorCriticEncoderGLAD(obs, GROUPS, 29, top_k=top_k, cnn_downsample=downsample).eval()
    packed = model.get_actor_obs(obs)
    encoded, weights = model._encode_terrain(packed)
    grid = obs["policy"][:, 3:].reshape(3, 21, 33, 3).permute(0, 3, 1, 2)
    tokens = model.map_cnn(grid).flatten(2).transpose(1, 2)
    scores = model.saliency_score(tokens).squeeze(-1)
    expected_indices = scores.topk(top_k, dim=-1).indices
    selected, indices = model._select_local_features(tokens)
    torch.testing.assert_close(indices, expected_indices)
    torch.testing.assert_close(selected, tokens.gather(1, indices[..., None].expand(-1, -1, 64)))
    global_weights = model.global_encoder(tokens).softmax(dim=1)
    torch.testing.assert_close(encoded[:, :64], (global_weights * tokens).sum(dim=1))
    torch.testing.assert_close(encoded[:, 128:], packed[:, :-2079])
    assert weights.shape == (3, 1, 187 if downsample else 693)
    support = torch.zeros_like(weights, dtype=torch.bool).scatter(2, indices[:, None], True)
    assert torch.count_nonzero(weights[~support]) == 0
    assert (weights[support] > 0).all()
    torch.testing.assert_close(weights.sum(-1), torch.ones(3, 1))
    actions, actual_weights = model.act_inference(obs)
    assert actions.shape == (3, 29)
    assert model.evaluate(obs).shape == (3, 1)
    torch.testing.assert_close(weights, actual_weights)


def test_training_selection_is_hard_stochastic_and_differentiable():
    model = ActorCriticEncoderGLAD(observations(), GROUPS, 29)
    tokens = torch.randn(3, 187, 64, requires_grad=True)
    selected, indices = model._select_local_features(tokens)
    torch.testing.assert_close(
        selected, tokens.gather(1, indices[..., None].expand(-1, -1, 64)), rtol=0, atol=0,
    )
    assert all(len(row.unique()) == 32 for row in indices)
    selected.square().mean().backward()
    assert model.saliency_score.weight.grad.abs().sum() > 0
    assert torch.isfinite(tokens.grad).all()
    # Rollouts run under inference_mode while the policy remains in train mode.
    with torch.inference_mode():
        _, first = model._select_local_features(tokens)
        _, second = model._select_local_features(tokens)
    assert not torch.equal(first, second)
    model.eval()
    _, first = model._select_local_features(tokens)
    _, second = model._select_local_features(tokens)
    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_inference_is_deterministic_and_critic_does_not_read_history():
    obs = observations()
    snapshot = obs.clone()
    model = ActorCriticEncoderGLAD(obs, GROUPS, 29).eval()
    with torch.no_grad():
        baseline, weights = model.act_inference(obs)
        value = model.evaluate(obs)
        repeated = model.act_inference(obs)
        torch.testing.assert_close(baseline, repeated[0], rtol=0, atol=0)
        torch.testing.assert_close(weights, repeated[1], rtol=0, atol=0)
        early = obs.clone()
        early["proprio_history"][:, :20] += 5
        assert not torch.allclose(baseline, model.act_inference(early)[0])
        torch.testing.assert_close(value, model.evaluate(early), rtol=0, atol=0)
        # Disable the long encoder to isolate the short-history bypass.
        for parameter in model.history_encoder.parameters():
            parameter.zero_()
        baseline = model.act_inference(obs)[0]
        torch.testing.assert_close(baseline, model.act_inference(early)[0], rtol=0, atol=0)
        recent = obs.clone()
        recent["proprio_history"][:, -4:] += 5
        assert not torch.allclose(baseline, model.act_inference(recent)[0])
        torch.testing.assert_close(value, model.evaluate(recent), rtol=0, atol=0)
    for key in obs.keys():
        torch.testing.assert_close(obs[key], snapshot[key], rtol=0, atol=0)


@pytest.mark.parametrize("kwargs", [
    {"top_k": 0}, {"top_k": 188}, {"top_k": True}, {"top_k": 1.5},
    {"gumbel_temperature": 0}, {"gumbel_temperature": float("nan")},
    {"gumbel_temperature": float("inf")}, {"attach_global": False},
    {"actor_obs_normalization": True}, {"critic_obs_normalization": True},
])
def test_reject_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        ActorCriticEncoderGLAD(observations(), GROUPS, 29, **kwargs)


def test_checkpoint_architecture_and_selection_validation():
    obs = observations()
    model = ActorCriticEncoderGLAD(obs, GROUPS, 29)
    lsio = ActorCriticEncoderLSIO(obs, GROUPS, 29, attach_global=True)
    for state in (lsio.state_dict(), {}):
        with pytest.raises(RuntimeError, match="GLAD checkpoint"):
            model.load_state_dict(state)
    for kwargs in ({"top_k": 16}, {"gumbel_temperature": 0.5}, {"long_history_length": 67}):
        changed = ActorCriticEncoderGLAD(observations(history=kwargs.get("long_history_length", 66)), GROUPS, 29, **kwargs)
        with pytest.raises(RuntimeError, match="GLAD checkpoint"):
            changed.load_state_dict(model.state_dict())
    with pytest.raises(RuntimeError, match="LSIO checkpoint"):
        lsio.load_state_dict(model.state_dict())


def glad_runner_config():
    cfg = runner_config()
    cfg["policy"] = {"class_name": "ActorCriticEncoderGLAD"}
    return cfg


def test_cpu_ppo_save_restore_and_resume(tmp_path):
    env = SyntheticEnv()
    runner = OnPolicyRunner(env, glad_runner_config(), log_dir=str(tmp_path / "first"))
    scorers = (runner.alg.policy.global_encoder, runner.alg.policy.saliency_score)
    before = [layer.weight.detach().clone() for layer in scorers]
    runner.learn(2)
    assert env.reset_count > 0
    for weight, layer in zip(before, scorers):
        assert not torch.equal(weight, layer.weight)
    assert all(torch.isfinite(p).all() for p in runner.alg.policy.parameters())
    checkpoint = tmp_path / "glad.pt"
    runner.save(str(checkpoint))
    obs = env.get_observations()
    with torch.no_grad():
        expected = runner.get_inference_policy()(obs)
    restored = OnPolicyRunner(SyntheticEnv(), glad_runner_config(), log_dir=str(tmp_path / "resume"))
    restored.load(str(checkpoint), map_location="cpu")
    assert restored.current_learning_iteration == runner.current_learning_iteration
    original_optimizer = runner.alg.optimizer.state_dict()
    restored_optimizer = restored.alg.optimizer.state_dict()
    assert original_optimizer["state"]
    assert original_optimizer["param_groups"] == restored_optimizer["param_groups"]
    for key, state in original_optimizer["state"].items():
        for name, value in state.items():
            torch.testing.assert_close(value, restored_optimizer["state"][key][name])
    with torch.no_grad():
        actual = restored.get_inference_policy()(obs)
    for left, right in zip(expected, actual):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    restored.learn(1)
    assert all(torch.isfinite(p).all() for p in restored.alg.policy.parameters())
    runner.writer.close()
    restored.writer.close()


def test_train_play_registration_reuses_lsio_environment():
    # Execute actual registration calls without importing simulation extensions.
    tree = ast.parse((TASK / "29dof/__init__.py").read_text())
    tree.body = [node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
    registrations = {}
    namespace = {
        "__name__": "ame_locomotion.tasks.manager_based.ame_locomotion.29dof",
        "agents": SimpleNamespace(__name__="ame_locomotion.tasks.manager_based.ame_locomotion.agents"),
        "gym": SimpleNamespace(register=lambda **kwargs: registrations.update({kwargs["id"]: kwargs})),
    }
    exec(compile(tree, str(TASK / "29dof/__init__.py"), "exec"), namespace)
    for suffix in ("v0", "Play-v0"):
        glad = registrations[f"AME-G1-29DOF-GLAD-{suffix}"]
        lsio = registrations[f"AME-G1-29DOF-LSIO-{suffix}"]
        assert glad["entry_point"] == lsio["entry_point"]
        assert glad["kwargs"]["env_cfg_entry_point"] == lsio["kwargs"]["env_cfg_entry_point"]
        assert glad["kwargs"]["rsl_rl_cfg_entry_point"].endswith(":G1AMEGLADPPORunnerCfg")
    assert "AME-G1-29DOF-v0" in registrations
    assert "AME-G1-29DOF-Play-v0" in registrations


def test_native_runner_config_keeps_ppo_and_observation_settings():
    isaaclab_rl = pytest.importorskip("isaaclab_rl")
    package = ModuleType("_glad_rl_cfg")
    package.__path__ = [str(Path(isaaclab_rl.__file__).parent / "rsl_rl")]
    with patch.dict(sys.modules, {package.__name__: package}):
        rl_cfg = importlib.import_module("_glad_rl_cfg.rl_cfg")
        with patch.dict(sys.modules, {"isaaclab_rl.rsl_rl": rl_cfg}):
            agent = load_file("_glad_agent_cfg", TASK / "agents/ame_rsl_rl_ppo_cfg.py")
    old = agent.G1AMELSIOPPORunnerCfg().to_dict()
    new = agent.G1AMEGLADPPORunnerCfg().to_dict()
    assert new["experiment_name"] == "g1_ame_glad"
    expected = deepcopy(old)
    expected["experiment_name"] = "g1_ame_glad"
    expected["policy"].update(
        class_name="ActorCriticEncoderGLAD", attach_global=True, top_k=32, gumbel_temperature=1.0,
    )
    assert new == expected
    runner = OnPolicyRunner(SyntheticEnv(), new)
    with torch.no_grad():
        actions, attention = runner.get_inference_policy()(observations())
    assert actions.shape == (3, 29)
    assert attention.shape == (3, 1, 187)
