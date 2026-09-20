"""CPU gradient/state isolation and task/checkpoint integration for all ablations."""

import ast
import importlib
import sys
from copy import deepcopy
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from rsl_rl.modules import ActorCriticEncoder, ActorCriticEncoderGLAD, ActorCriticEncoderLSIO
from rsl_rl.runners import OnPolicyRunner
from test_lsio import GROUPS, SyntheticEnv, observations, runner_config
from test_lsio_history import TASK, load_file


VARIANTS = [
    pytest.param(ActorCriticEncoder, False, id="ame"),
    pytest.param(ActorCriticEncoder, True, id="ame2"),
    pytest.param(ActorCriticEncoderLSIO, False, id="lsio"),
    pytest.param(ActorCriticEncoderLSIO, True, id="ame2-lsio"),
    pytest.param(ActorCriticEncoderGLAD, True, id="glad"),
]
SHARED = ("map_cnn", "mha", "global_encoder", "query_projector", "saliency_score")


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.manual_seed(42)
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def model_observations(cls, batch=3):
    obs = observations(batch)
    if cls is ActorCriticEncoder:
        obs["policy"] = torch.randn(batch, 2175)
    return obs


def make_model(cls, attach_global, stop_grad=True):
    obs = model_observations(cls)
    return cls(obs, GROUPS, 29, attach_global=attach_global, critic_encoder_stop_grad=stop_grad), obs


def is_critic(name):
    return name.startswith("critic.")


def assert_same(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_same(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_same(a, b)
    else:
        assert left == right


@pytest.mark.parametrize("cls,attach_global", VARIANTS)
def test_critic_only_step_preserves_actor_and_buffers_with_existing_momentum(cls, attach_global):
    model, obs = make_model(cls, attach_global)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.01)
    # Populate shared/Actor optimizer state first: zero gradients would still
    # allow Adam momentum and weight decay to move these parameters.
    _, value = model.act_and_evaluate(obs)
    (model.action_mean.square().mean() + value.square().mean()).backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    assert optimizer.state[model.map_cnn[0].weight]
    model.eval()
    with torch.no_grad():
        expected_actor = model.act_inference(obs)
    before = deepcopy(model.state_dict())
    distribution = model.distribution
    model.train()
    value = model.evaluate(obs)
    assert_same(before, model.state_dict())  # Forward must not write any buffers.
    assert model.distribution is distribution
    value.square().mean().backward()
    for name, parameter in model.named_parameters():
        if is_critic(name):
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
        else:
            assert parameter.grad is None, name
    assert not model.critic_proprio_embedding.weight.requires_grad
    assert model.critic[0].weight.grad.abs().sum() > 0
    optimizer.step()
    after = model.state_dict()
    for name in before:
        if not is_critic(name):
            assert_same(before[name], after[name])
    assert not torch.equal(before["critic.0.weight"], after["critic.0.weight"])
    model.eval()
    with torch.no_grad():
        assert_same(expected_actor, model.act_inference(obs))


@pytest.mark.parametrize("cls,attach_global", VARIANTS)
def test_actor_trains_encoder_and_joint_gradient_equals_actor_only(cls, attach_global):
    actor_only, obs = make_model(cls, attach_global)
    joint = deepcopy(actor_only)
    rng = torch.get_rng_state()
    actor_only.act(obs)
    actor_only.action_mean.square().mean().backward()
    torch.set_rng_state(rng)
    _, value = joint.act_and_evaluate(obs)
    actor_loss = joint.action_mean.square().mean()
    buffers = deepcopy(dict(joint.named_buffers()))
    # Also preserve the Actor graph across a standalone bootstrap call. Its
    # temporary CNN buffers must not invalidate tensors saved for backward.
    bootstrap_value = joint.evaluate(obs)
    assert_same(buffers, dict(joint.named_buffers()))
    (actor_loss + value.square().mean() + bootstrap_value.square().mean()).backward()
    for name, parameter in actor_only.named_parameters():
        if not is_critic(name):
            assert_same(parameter.grad, joint.get_parameter(name).grad)
    for name in (*SHARED, "actor_proprio_embedding", "history_encoder"):
        module = getattr(actor_only, name, None)
        if module is not None:
            grads = [p.grad for p in module.parameters()]
            assert all(g is not None and torch.isfinite(g).all() for g in grads), name
            assert sum(g.abs().sum() for g in grads) > 0, name
    assert joint.critic_proprio_embedding.weight.grad is None
    assert joint.critic[0].weight.grad.abs().sum() > 0


@pytest.mark.parametrize("cls,attach_global", VARIANTS)
@pytest.mark.parametrize("training", [False, True])
def test_actor_forward_and_attention_parity_without_extra_parameters(cls, attach_global, training):
    original, obs = make_model(cls, attach_global, stop_grad=False)
    isolated = cls(obs, GROUPS, 29, attach_global=attach_global, critic_encoder_stop_grad=True)
    isolated.load_state_dict(original.state_dict())
    original.train(training)
    isolated.train(training)
    assert_same(original.state_dict(), isolated.state_dict())
    assert sum(p.numel() for p in original.parameters()) == sum(p.numel() for p in isolated.parameters())
    rng = torch.get_rng_state()
    expected = original.act_inference(obs)
    torch.set_rng_state(rng)
    actual = isolated.act_inference(obs)
    assert_same(expected, actual)
    assert_same(dict(original.named_buffers()), dict(isolated.named_buffers()))
    # Unlike the original Critic, the ablation consumes Actor features. Its
    # standalone value call must neither touch the distribution nor BN state.
    isolated.evaluate(obs)
    assert isolated.distribution is None
    assert_same(dict(original.named_buffers()), dict(isolated.named_buffers()))
    for name, buffer in isolated.named_buffers():
        if name.endswith("num_batches_tracked"):
            assert buffer.item() == (1 if training else 0)


@pytest.mark.parametrize("cls,attach_global", VARIANTS)
@pytest.mark.parametrize("training", [False, True])
def test_joint_forward_reuses_exact_actor_features_and_only_value_head_gets_gradients(cls, attach_global, training):
    model, obs = make_model(cls, attach_global)
    model.train(training)
    inputs = {}

    def capture(name):
        def hook(module, args):
            inputs[name] = args[0].detach().clone()
        return hook

    actor_hook = model.actor.register_forward_pre_hook(capture("actor"))
    critic_hook = model.critic.register_forward_pre_hook(capture("critic"))
    try:
        with patch.object(model, "_encode_terrain", wraps=model._encode_terrain) as encode:
            with patch.object(model.critic_proprio_embedding, "forward", side_effect=AssertionError("unused Query")):
                actions, value = model.act_and_evaluate(obs)
            assert encode.call_count == 1
    finally:
        actor_hook.remove()
        critic_hook.remove()
    assert actions.shape == (3, 29)
    assert value.shape == (3, 1)
    terrain_width = 128 if attach_global else 64
    assert inputs["critic"].shape == (3, terrain_width + 99)
    assert_same(inputs["critic"][:, :terrain_width], inputs["actor"][:, :terrain_width])
    assert_same(inputs["critic"][:, terrain_width:], obs["critic"][:, :99])
    value.square().mean().backward()
    for name, parameter in model.named_parameters():
        if is_critic(name):
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        else:
            assert parameter.grad is None, name


@pytest.mark.parametrize("cls,attach_global", VARIANTS)
def test_explicit_feature_input_is_detached_without_reencoding(cls, attach_global):
    model, obs = make_model(cls, attach_global)
    features = torch.randn(3, 128 if attach_global else 64, requires_grad=True)
    before = deepcopy(model.state_dict())
    with patch.object(model, "_encode_terrain", side_effect=AssertionError("must reuse features")):
        value = model.evaluate(obs, actor_terrain_features=features)
    assert_same(before, model.state_dict())
    assert_same(value, model.critic(torch.cat((features.detach(), obs["critic"][:, :99]), dim=-1)))
    value.square().mean().backward()
    assert features.grad is None
    assert model.critic[0].weight.grad.abs().sum() > 0
    with pytest.raises(ValueError, match="must have shape"):
        model.evaluate(obs, actor_terrain_features=features[:, :-1])


@pytest.mark.parametrize("cls,attach_global", VARIANTS)
@pytest.mark.parametrize("training", [False, True])
def test_standalone_value_uses_actor_observations_and_preserves_state(cls, attach_global, training):
    model, obs = make_model(cls, attach_global)
    model.train(training)
    rng = torch.get_rng_state()
    # Obtain an independent reference through the ordinary Actor encoder.
    reference = deepcopy(model)
    with torch.no_grad():
        encoded, _ = reference._encode_terrain(reference.get_actor_obs(obs))
        features = encoded[:, :-reference.actor_proprio_dim]
        expected = reference.critic(torch.cat((features, obs["critic"][:, :99]), dim=-1))
    before = deepcopy(model.state_dict())
    torch.set_rng_state(rng)
    value = model.evaluate(obs)
    torch.testing.assert_close(expected, value)
    assert_same(before, model.state_dict())
    assert model.distribution is None
    # Equal RNG avoids mistaking GLAD selection randomness for input dependence.
    def evaluate_same_rng(changed):
        torch.set_rng_state(rng)
        return model.evaluate(changed)

    changed = obs.clone()
    changed["critic"][:, 99:] += torch.randn_like(changed["critic"][:, 99:]) * 5
    assert_same(value, evaluate_same_rng(changed))
    changed = obs.clone()
    changed["policy"][:, -2079:] = torch.randn_like(changed["policy"][:, -2079:])
    assert not torch.allclose(value, evaluate_same_rng(changed))
    changed = obs.clone()
    changed["critic"][:, :99] += 1
    assert not torch.allclose(value, evaluate_same_rng(changed))
    if cls is not ActorCriticEncoder:
        changed = obs.clone()
        changed["proprio_history"] += 2
        # Randomly initialized attention can be almost uniform. Test actual
        # dependence, without demanding a minimum effect size before training.
        assert not torch.equal(value, evaluate_same_rng(changed))
    # The latest observations are encoded on every standalone call, with no
    # stale features left behind by the previous changed inputs.
    assert_same(value, evaluate_same_rng(obs))


def test_glad_joint_and_standalone_calls_each_select_once_without_sampling_bootstrap_actions():
    model, obs = make_model(ActorCriticEncoderGLAD, True)
    with patch.object(model, "_select_local_features", wraps=model._select_local_features) as select:
        model.act_and_evaluate(obs)
        assert select.call_count == 1
        distribution = model.distribution
        state = deepcopy(model.state_dict())
        with patch.object(distribution, "sample", side_effect=AssertionError("value-only call")):
            model.evaluate(obs)
        assert select.call_count == 2
        assert model.distribution is distribution
        assert_same(state, model.state_dict())


@pytest.mark.parametrize("cls,attach_global", VARIANTS)
def test_inference_mode_critic_isolates_rollout_and_bootstrap_buffers(cls, attach_global):
    model, obs = make_model(cls, attach_global)
    model.train()
    buffers = deepcopy(dict(model.named_buffers()))
    with torch.inference_mode():
        for _ in range(3):
            assert torch.isfinite(model.evaluate(obs)).all()
    assert_same(buffers, dict(model.named_buffers()))
    # Temporary inference tensors must not remain installed on the module.
    model.act(obs)
    (model.action_mean.square().mean() + model.evaluate(obs).square().mean()).backward()
    assert model.map_cnn[0].weight.grad.abs().sum() > 0


class AblationEnv(SyntheticEnv):
    def __init__(self, cls):
        super().__init__()
        self.obs = model_observations(cls, self.num_envs)


def config(cls, attach_global, stop_grad=True):
    cfg = runner_config()
    cfg["policy"] = {
        "class_name": cls.__name__, "attach_global": attach_global,
        "critic_encoder_stop_grad": stop_grad,
    }
    return cfg


@pytest.mark.parametrize("cls,attach_global", VARIANTS)
def test_ppo_save_resume_and_critic_bn_call_counts(cls, attach_global, tmp_path):
    env = AblationEnv(cls)
    runner = OnPolicyRunner(env, config(cls, attach_global), log_dir=str(tmp_path / "first"))
    before = runner.alg.policy.map_cnn[0].weight.detach().clone()
    model = runner.alg.policy
    with patch.object(model, "act_and_evaluate", wraps=model.act_and_evaluate) as joint:
        with patch.object(model, "_encode_terrain", wraps=model._encode_terrain) as encode:
            with patch.object(model, "evaluate", wraps=model.evaluate) as evaluate:
                runner.learn(2)
            assert evaluate.call_count == 14
            assert sum(call.kwargs.get("actor_terrain_features") is not None for call in evaluate.call_args_list) == 12
        assert encode.call_count == 14  # 12 joint calls + 2 standalone bootstraps.
    assert joint.call_count == 12
    assert not torch.equal(before, runner.alg.policy.map_cnn[0].weight)
    # Each iteration: 4 rollout Actor calls + 2 minibatch Actor calls.
    # Critic rollout, bootstrap, and minibatch calls must add no BN updates.
    for name, buffer in runner.alg.policy.named_buffers():
        if name.endswith("num_batches_tracked"):
            assert buffer.item() == 12
    path = tmp_path / "ablation.pt"
    runner.save(str(path), infos={"test": True})
    saved = torch.load(path, weights_only=False)
    assert saved["critic_encoder_stop_grad"] is True
    assert saved["critic_feature_source"] == "actor"
    if "_extra_state" in saved["model_state_dict"]:
        assert "critic_encoder_stop_grad" not in saved["model_state_dict"]["_extra_state"]
        assert "critic_feature_source" not in saved["model_state_dict"]["_extra_state"]
    restored = OnPolicyRunner(AblationEnv(cls), config(cls, attach_global), log_dir=str(tmp_path / "resume"))
    assert restored.load(str(path), map_location="cpu") == {"test": True}
    assert restored.current_learning_iteration == runner.current_learning_iteration
    assert_same(runner.alg.policy.state_dict(), restored.alg.policy.state_dict())
    assert_same(runner.alg.optimizer.state_dict(), restored.alg.optimizer.state_dict())
    with torch.no_grad():
        obs = env.get_observations()
        assert_same(runner.get_inference_policy()(obs), restored.get_inference_policy()(obs))
    resumed_before = restored.alg.policy.map_cnn[0].weight.detach().clone()
    restored.learn(1)
    assert not torch.equal(resumed_before, restored.alg.policy.map_cnn[0].weight)
    assert all(torch.isfinite(p).all() for p in restored.alg.policy.parameters())
    runner.writer.close()
    restored.writer.close()


@pytest.mark.parametrize("cls,attach_global", VARIANTS)
@pytest.mark.parametrize("stop_grad", [False, True])
@pytest.mark.parametrize("load_optimizer", [False, True])
def test_checkpoint_modes_validate_before_mutation(cls, attach_global, stop_grad, load_optimizer, tmp_path):
    runner = OnPolicyRunner(AblationEnv(cls), config(cls, attach_global, stop_grad), log_dir=str(tmp_path / "run"))
    runner.learn(1)  # Include initialized Adam state in the no-mutation check.
    before_model = deepcopy(runner.alg.policy.state_dict())
    before_optimizer = deepcopy(runner.alg.optimizer.state_dict())
    before_iteration = runner.current_learning_iteration
    checkpoint = {
        "model_state_dict": deepcopy(before_model),
        "optimizer_state_dict": deepcopy(before_optimizer), "iter": 999, "infos": None,
    }
    checkpoint["model_state_dict"]["critic.0.weight"].add_(1)
    # A missing flag represents an actual legacy checkpoint, not an opt-in.
    if not stop_grad:
        checkpoint["critic_encoder_stop_grad"] = True
    path = tmp_path / "opposite_mode.pt"
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="critic_encoder_stop_grad mode mismatch"):
        runner.load(str(path), load_optimizer=load_optimizer)
    assert_same(before_model, runner.alg.policy.state_dict())
    assert_same(before_optimizer, runner.alg.optimizer.state_dict())
    assert runner.current_learning_iteration == before_iteration
    runner.writer.close()


@pytest.mark.parametrize("cls,attach_global", VARIANTS)
@pytest.mark.parametrize("load_optimizer", [False, True])
def test_reject_old_stop_grad_and_wrong_feature_sources_before_mutation(cls, attach_global, load_optimizer, tmp_path):
    runner = OnPolicyRunner(AblationEnv(cls), config(cls, attach_global), log_dir=str(tmp_path / "run"))
    runner.learn(1)
    path = tmp_path / "ablation.pt"
    runner.save(str(path))
    checkpoint = torch.load(path, weights_only=False)
    before_model = deepcopy(runner.alg.policy.state_dict())
    before_optimizer = deepcopy(runner.alg.optimizer.state_dict())
    before_iteration = runner.current_learning_iteration
    checkpoint["model_state_dict"]["critic.0.weight"].add_(1)
    checkpoint["iter"] = 999
    for source in (None, "critic", "unknown"):
        if source is None:
            del checkpoint["critic_feature_source"]  # Old CriticStopGrad format.
        else:
            checkpoint["critic_feature_source"] = source
        torch.save(checkpoint, path)
        with pytest.raises(ValueError, match="critic_feature_source mismatch"):
            runner.load(str(path), load_optimizer=load_optimizer)
        assert_same(before_model, runner.alg.policy.state_dict())
        assert_same(before_optimizer, runner.alg.optimizer.state_dict())
        assert runner.current_learning_iteration == before_iteration
    runner.writer.close()


@pytest.mark.parametrize("cls,attach_global", VARIANTS)
def test_original_runner_still_accepts_legacy_and_explicit_false(cls, attach_global, tmp_path):
    cfg = config(cls, attach_global, False)
    del cfg["policy"]["critic_encoder_stop_grad"]  # Existing task defaults.
    runner = OnPolicyRunner(AblationEnv(cls), cfg, log_dir=str(tmp_path / "run"))
    assert runner.alg.policy.critic_encoder_stop_grad is False
    runner.learn(1)
    path = tmp_path / "legacy.pt"
    runner.save(str(path))
    saved = torch.load(path, weights_only=False)
    assert "critic_encoder_stop_grad" not in saved
    assert "critic_feature_source" not in saved
    for explicit_false in (False, True):
        if explicit_false:
            saved["critic_encoder_stop_grad"] = False
            torch.save(saved, path)
        runner.load(str(path))
        assert_same(saved["model_state_dict"], runner.alg.policy.state_dict())
    runner.writer.close()


def test_all_train_play_registrations_reuse_original_environments():
    tree = ast.parse((TASK / "29dof/__init__.py").read_text())
    tree.body = [node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
    registrations = {}
    namespace = {
        "__name__": "ame_locomotion.tasks.manager_based.ame_locomotion.29dof",
        "agents": SimpleNamespace(__name__="ame_locomotion.tasks.manager_based.ame_locomotion.agents"),
        "gym": SimpleNamespace(register=lambda **kw: registrations.update({kw["id"]: kw})),
    }
    exec(compile(tree, str(TASK / "29dof/__init__.py"), "exec"), namespace)
    for variant, runner_name in (("", "G1AME"), ("-LSIO", "G1AMELSIO"), ("-GLAD", "G1AMEGLAD")):
        for suffix in ("v0", "Play-v0"):
            base = registrations[f"AME-G1-29DOF{variant}-{suffix}"]
            new = registrations[f"AME-G1-29DOF{variant}-CriticStopGrad-{suffix}"]
            assert new["entry_point"] == base["entry_point"]
            assert new["disable_env_checker"] == base["disable_env_checker"]
            assert new["kwargs"]["env_cfg_entry_point"] == base["kwargs"]["env_cfg_entry_point"]
            assert new["kwargs"]["rsl_rl_cfg_entry_point"].endswith(f":{runner_name}CriticStopGradPPORunnerCfg")
    for suffix in ("v0", "Play-v0"):
        base = registrations[f"AME-G1-29DOF-GLAD-{suffix}"]
        new = registrations[f"AME-G1-29DOF-GLAD-CriticCleanStopGrad-{suffix}"]
        assert new["kwargs"]["env_cfg_entry_point"] == base["kwargs"]["env_cfg_entry_point"]
        assert new["kwargs"]["rsl_rl_cfg_entry_point"].endswith(":G1AMEGLADCriticCleanStopGradPPORunnerCfg")


def test_native_configs_only_change_mode_and_experiment():
    isaaclab_rl = pytest.importorskip("isaaclab_rl")
    # Keep native extensions imported by utils in sys.modules across the
    # temporary import aliases. Some cannot safely be imported a second time.
    importlib.import_module("isaaclab.utils")
    package = ModuleType("_critic_stop_grad_rl_cfg")
    package.__path__ = [str(Path(isaaclab_rl.__file__).parent / "rsl_rl")]
    with patch.dict(sys.modules, {package.__name__: package}):
        rl_cfg = importlib.import_module(f"{package.__name__}.rl_cfg")
        with patch.dict(sys.modules, {"isaaclab_rl.rsl_rl": rl_cfg}):
            agent = load_file("_critic_stop_grad_agent_cfg", TASK / "agents/ame_rsl_rl_ppo_cfg.py")
    for name, experiment in (("G1AME", "g1_ame"), ("G1AMELSIO", "g1_ame_lsio"), ("G1AMEGLAD", "g1_ame_glad")):
        base_cls = getattr(agent, f"{name}PPORunnerCfg")
        new_cls = getattr(agent, f"{name}CriticStopGradPPORunnerCfg")
        assert issubclass(new_cls, base_cls)
        expected = base_cls().to_dict()
        expected["experiment_name"] = f"{experiment}_critic_stop_grad"
        expected["policy"]["critic_encoder_stop_grad"] = True
        expected["policy"].setdefault("attach_global", False)
        cfg = new_cls()
        assert cfg.to_dict() == expected
        assert cfg.resume is False
        # The AME2 switch is a declared config field accepted by from_dict/Hydra.
        cfg.from_dict({"policy": {"attach_global": True}})
        assert cfg.policy.attach_global is True
        assert base_cls().to_dict()["policy"].get("critic_encoder_stop_grad", False) is False
    expected = agent.G1AMEGLADPPORunnerCfg().to_dict()
    expected["experiment_name"] = "g1_ame_glad_critic_clean_stop_grad"
    expected["policy"].update(critic_encoder_stop_grad=True, critic_feature_source="critic")
    clean = agent.G1AMEGLADCriticCleanStopGradPPORunnerCfg()
    assert clean.to_dict() == expected
    assert clean.resume is False
    assert isinstance(clean, agent.G1AMEGLADPPORunnerCfg)
