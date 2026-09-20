"""Clean Critic observations, frozen Query projection and detached terrain output."""

from copy import deepcopy
from unittest.mock import patch

import pytest
import torch

from rsl_rl.modules import ActorCriticEncoderGLAD
from rsl_rl.runners import OnPolicyRunner
from test_critic_stop_grad import AblationEnv, SHARED, assert_same, config
from test_lsio import GROUPS, observations


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.manual_seed(42)
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def make_model():
    obs = observations()
    model = ActorCriticEncoderGLAD(
        obs, GROUPS, 29, critic_encoder_stop_grad=True, critic_feature_source="critic",
    )
    return model, obs


def clean_config():
    cfg = config(ActorCriticEncoderGLAD, True)
    cfg["policy"]["critic_feature_source"] = "critic"
    return cfg


@pytest.mark.parametrize("training", [False, True])
def test_matches_normal_critic_encoding_and_uses_private_query(training):
    model, obs = make_model()
    normal = ActorCriticEncoderGLAD(obs, GROUPS, 29)
    normal.load_state_dict(model.state_dict())
    model.train(training)
    normal.train(training)
    assert model.critic[0].in_features == 227
    assert not model.critic_proprio_embedding.weight.requires_grad
    inputs = []
    hook = model.critic.register_forward_pre_hook(lambda module, args: inputs.append(args[0].detach().clone()))
    before = deepcopy(model.state_dict())
    rng = torch.get_rng_state()
    with patch.object(model.critic_proprio_embedding, "forward", wraps=model.critic_proprio_embedding.forward) as query:
        value = model.evaluate(obs)
    hook.remove()
    assert query.call_count == 1
    assert_same(query.call_args.args[0], obs["critic"][:, :99])
    assert_same(before, model.state_dict())
    torch.set_rng_state(rng)
    encoded, _ = normal._encode_terrain(normal.get_critic_obs(obs), role="critic")
    assert_same(inputs[0], encoded.detach())
    assert_same(value, normal.critic(encoded))
    assert_same(inputs[0][:, 128:], obs["critic"][:, :99])
    # The frozen projection is active, rather than bypassed or replaced by Actor Query.
    with torch.no_grad():
        model.critic_proprio_embedding.weight.add_(1)
    torch.set_rng_state(rng)
    assert not torch.equal(value, model.evaluate(obs))


def test_value_only_with_existing_adam_momentum_preserves_all_encoder_state():
    model, obs = make_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.01)
    # Seed momentum even on the private projection, as a stronger isolation check.
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    model.eval()
    expected = model.act_inference(obs)
    model.act(obs)
    distribution = model.distribution
    before = deepcopy(model.state_dict())
    model.train()
    with patch.object(model, "get_actor_obs", side_effect=AssertionError("Critic must not encode Actor")):
        model.evaluate(obs).square().mean().backward()
    assert model.distribution is distribution
    assert_same(before, model.state_dict())
    for name, parameter in model.named_parameters():
        if name.startswith("critic."):
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        else:
            assert parameter.grad is None, name
    assert model.critic[0].weight.grad.abs().sum() > 0
    optimizer.step()
    for name, value in model.state_dict().items():
        if not name.startswith("critic."):
            assert_same(before[name], value)
    model.eval()
    assert_same(expected, model.act_inference(obs))


@pytest.mark.parametrize("training", [False, True])
def test_value_depends_only_on_critic_observations(training):
    model, obs = make_model()
    model.train(training)
    rng = torch.get_rng_state()
    def evaluate(data):
        torch.set_rng_state(rng)
        return model.evaluate(data)
    expected = evaluate(obs)
    for key in ("policy", "proprio_history"):
        changed = obs.clone()
        changed[key] += torch.randn_like(changed[key]) * 3
        assert_same(expected, evaluate(changed))
    for section in (slice(99, None), slice(None, 99)):
        changed = obs.clone()
        changed["critic"][:, section] += torch.randn_like(changed["critic"][:, section]) * 3
        assert not torch.equal(expected, evaluate(changed))
    assert_same(expected, evaluate(obs.select("critic")))
    with pytest.raises(ValueError, match="critic_feature_source"):
        model.evaluate(obs, actor_terrain_features=torch.zeros(3, 128))


def test_joint_gradients_and_pending_actor_graph_are_preserved():
    model, obs = make_model()
    actor_only = deepcopy(model)
    rng = torch.get_rng_state()
    with patch.object(model, "_encode_terrain", wraps=model._encode_terrain) as encode:
        with patch.object(model, "_select_local_features", wraps=model._select_local_features) as select:
            _, value = model.act_and_evaluate(obs)
    assert [call.kwargs.get("role", "actor") for call in encode.call_args_list] == ["actor", "critic"]
    assert select.call_count == 2
    buffers = deepcopy(dict(model.named_buffers()))
    # Bootstrap uses an independent Critic encoding, without invalidating Actor's graph.
    distribution = model.distribution
    with torch.inference_mode():
        model.evaluate(obs)
    assert model.distribution is distribution
    assert_same(buffers, dict(model.named_buffers()))
    (model.action_mean.square().mean() + value.square().mean()).backward()
    torch.set_rng_state(rng)
    actor_only.act(obs)
    actor_only.action_mean.square().mean().backward()
    for name, parameter in model.named_parameters():
        if name.startswith("critic."):
            continue
        expected = dict(actor_only.named_parameters())[name].grad
        if expected is None:
            assert parameter.grad is None, name
        else:
            assert_same(parameter.grad, expected)
    for layer in (*SHARED, "history_encoder", "actor_proprio_embedding", "actor"):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in getattr(model, layer).parameters()), layer


def test_ppo_rollout_bootstrap_save_restore_and_continue(tmp_path):
    env = AblationEnv(ActorCriticEncoderGLAD)
    runner = OnPolicyRunner(env, clean_config(), log_dir=str(tmp_path / "first"))
    model = runner.alg.policy
    query = deepcopy(model.critic_proprio_embedding.state_dict())
    before = model.map_cnn[0].weight.detach().clone()
    with patch.object(model, "_encode_terrain", wraps=model._encode_terrain) as encode:
        runner.learn(2)
    assert sum(c.kwargs.get("role", "actor") == "actor" for c in encode.call_args_list) == 12
    assert sum(c.kwargs.get("role") == "critic" for c in encode.call_args_list) == 14
    for name, buffer in model.named_buffers():
        if name.endswith("num_batches_tracked"):
            assert buffer.item() == 12
    assert_same(query, model.critic_proprio_embedding.state_dict())
    assert not torch.equal(before, model.map_cnn[0].weight)
    path = tmp_path / "clean.pt"
    runner.save(str(path))
    saved = torch.load(path, weights_only=False)
    assert saved["critic_encoder_stop_grad"] is True
    assert saved["critic_feature_source"] == "critic"
    assert "critic_feature_source" not in saved["model_state_dict"]["_extra_state"]
    restored = OnPolicyRunner(env, clean_config(), log_dir=str(tmp_path / "restored"))
    restored.load(str(path))
    assert_same(model.state_dict(), restored.alg.policy.state_dict())
    assert_same(runner.alg.optimizer.state_dict(), restored.alg.optimizer.state_dict())
    assert restored.current_learning_iteration == runner.current_learning_iteration
    with torch.no_grad():
        assert_same(runner.get_inference_policy()(env.obs), restored.get_inference_policy()(env.obs))
    restored.load(str(path), load_optimizer=False)
    restored.learn(1)
    assert_same(query, restored.alg.policy.critic_proprio_embedding.state_dict())
    assert not torch.equal(model.map_cnn[0].weight, restored.alg.policy.map_cnn[0].weight)
    runner.writer.close()
    restored.writer.close()


@pytest.mark.parametrize("load_optimizer", [False, True])
@pytest.mark.parametrize("source", ["actor", None, "unknown", "normal"])
def test_checkpoint_rejection_precedes_any_mutation(tmp_path, source, load_optimizer):
    runner = OnPolicyRunner(AblationEnv(ActorCriticEncoderGLAD), clean_config(), log_dir=str(tmp_path / "run"))
    runner.learn(1)
    path = tmp_path / "checkpoint.pt"
    runner.save(str(path))
    saved = torch.load(path, weights_only=False)
    before = deepcopy(runner.alg.policy.state_dict()), deepcopy(runner.alg.optimizer.state_dict()), runner.current_learning_iteration
    saved["model_state_dict"]["critic.0.weight"].add_(1)
    saved["iter"] = 999
    if source == "normal":
        del saved["critic_encoder_stop_grad"]
        del saved["critic_feature_source"]
    elif source is None:
        del saved["critic_feature_source"]
    else:
        saved["critic_feature_source"] = source
    torch.save(saved, path)
    with pytest.raises(ValueError, match="mismatch"):
        runner.load(str(path), load_optimizer=load_optimizer)
    assert_same(before[0], runner.alg.policy.state_dict())
    assert_same(before[1], runner.alg.optimizer.state_dict())
    assert before[2] == runner.current_learning_iteration
    runner.writer.close()


@pytest.mark.parametrize("stop_grad,source,expected", [(False, None, "critic"), (True, None, "actor"), (True, "critic", "critic")])
def test_source_defaults(stop_grad, source, expected):
    model = ActorCriticEncoderGLAD(observations(), GROUPS, 29, critic_encoder_stop_grad=stop_grad, critic_feature_source=source)
    assert model.critic_feature_source == expected


@pytest.mark.parametrize("stop_grad,source", [(False, "actor"), (True, "bad"), (False, "bad")])
def test_invalid_source(stop_grad, source):
    with pytest.raises(ValueError):
        ActorCriticEncoderGLAD(observations(), GROUPS, 29, critic_encoder_stop_grad=stop_grad, critic_feature_source=source)
