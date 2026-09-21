"""Clean history through the Actor encoder; run with PYTHONPATH=rsl_rl:tests."""

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
from test_critic_stop_grad import SHARED, assert_same, config
from test_lsio import GROUPS, SyntheticEnv, observations
from test_lsio_history import TASK, NAMES, WIDTHS, configured_source, load_file, native, read_field


GROUPS_CLEAN = {**GROUPS, "clean_history": ["clean_proprio_history"]}


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.manual_seed(42)
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def clean_obs(batch=3):
    obs = observations(batch)
    obs["clean_proprio_history"] = torch.randn(batch, 66, 93)
    return obs


def make_model():
    obs = clean_obs()
    model = ActorCriticEncoderGLAD(
        obs, GROUPS_CLEAN, 29, critic_encoder_stop_grad=True, critic_feature_source="actor_clean",
    )
    return model, obs


def clean_config():
    cfg = config(ActorCriticEncoderGLAD, True)
    cfg["obs_groups"] = deepcopy(GROUPS_CLEAN)
    cfg["policy"]["critic_feature_source"] = "actor_clean"
    return cfg


@pytest.mark.parametrize("training", [False, True])
def test_explicit_clean_actor_reference_and_value_input(training):
    model, obs = make_model()
    model.train(training)
    # Reference uses the public Actor input path, replacing only its inputs.
    clean = obs.clone()
    clean["proprio_history"] = obs["clean_proprio_history"].clone()
    clean["policy"] = torch.cat((obs["critic"][:, 9:12], obs["critic"][:, 99:]), -1)
    reference = deepcopy(model)
    rng = torch.get_rng_state()
    with torch.no_grad():
        encoded, _ = reference._encode_terrain(reference.get_actor_obs(clean))
    expected = torch.cat((encoded[:, :128], obs["critic"][:, :99]), -1)
    before = deepcopy(model.state_dict())
    inputs = []
    hook = model.critic.register_forward_pre_hook(lambda module, args: inputs.append(args[0].detach().clone()))
    torch.set_rng_state(rng)
    with patch.object(model.critic_proprio_embedding, "forward", side_effect=AssertionError("unused Critic projection")):
        with patch.object(model.actor_proprio_embedding, "forward", wraps=model.actor_proprio_embedding.forward) as project:
            value = model.evaluate(obs)
    hook.remove()
    assert project.call_count == 1
    assert project.call_args.args[0].shape == (3, 519)
    assert_same(inputs[0], expected)
    assert_same(value, model.critic(expected))
    assert_same(before, model.state_dict())
    assert model.distribution is None
    assert not model.critic_proprio_embedding.weight.requires_grad


@pytest.mark.parametrize("training", [False, True])
def test_dependencies_and_no_feature_cache(training):
    model, obs = make_model()
    model.train(training)
    rng = torch.get_rng_state()
    def evaluate(data):
        torch.set_rng_state(rng)
        return model.evaluate(data)
    expected = evaluate(obs)
    for key in ("policy", "proprio_history"):
        changed = obs.clone()
        changed[key] += 5
        assert_same(expected, evaluate(changed))
    for key, selection in (("critic", slice(99, None)), ("critic", slice(0, 99)),
                           ("clean_proprio_history", slice(None))):
        changed = obs.clone()
        changed[key][:, selection] += torch.randn_like(changed[key][:, selection]) * 3
        assert not torch.equal(expected, evaluate(changed))
    # Long-history-only changes influence the clean Query as well.
    changed = obs.clone()
    changed["clean_proprio_history"][:, :20] += 3
    assert not torch.equal(expected, evaluate(changed))
    assert_same(expected, evaluate(obs))
    with pytest.raises(ValueError, match="critic_feature_source"):
        model.evaluate(obs, actor_terrain_features=torch.zeros(3, 128))


def test_value_only_preserves_actor_parameters_buffers_and_distribution_with_adam_momentum():
    model, obs = make_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.01)
    _, value = model.act_and_evaluate(obs)
    (model.action_mean.square().mean() + value.square().mean()).backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    assert optimizer.state[model.history_encoder[0].weight]
    model.eval()
    with torch.no_grad():
        expected_actor = model.act_inference(obs)
    before = deepcopy(model.state_dict())
    distribution = model.distribution
    model.train()
    with patch.object(model.actor, "forward", side_effect=AssertionError("no action MLP")):
        with patch.object(distribution, "sample", side_effect=AssertionError("no action sampling")):
            model.evaluate(obs).square().mean().backward()
    assert_same(before, model.state_dict())
    assert model.distribution is distribution
    for name, parameter in model.named_parameters():
        if name.startswith("critic."):
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        else:
            assert parameter.grad is None, name
    optimizer.step()
    for name, value in model.state_dict().items():
        if not name.startswith("critic."):
            assert_same(before[name], value)
    assert not torch.equal(before["critic.0.weight"], model.critic[0].weight)
    model.eval()
    with torch.no_grad():
        assert_same(expected_actor, model.act_inference(obs))


def test_joint_gradient_actor_graph_and_inference_bootstrap():
    model, obs = make_model()
    actor_only = deepcopy(model)
    rng = torch.get_rng_state()
    actor_only.act(obs)
    actor_only.action_mean.square().mean().backward()
    torch.set_rng_state(rng)
    with patch.object(model, "_select_local_features", wraps=model._select_local_features) as select:
        _, value = model.act_and_evaluate(obs)
        assert select.call_count == 2
        buffers = deepcopy(dict(model.named_buffers()))
        with torch.inference_mode():
            model.evaluate(obs)
        bootstrap = model.evaluate(obs)
        assert select.call_count == 4
    assert_same(buffers, dict(model.named_buffers()))
    (model.action_mean.square().mean() + value.square().mean() + bootstrap.square().mean()).backward()
    for name, parameter in actor_only.named_parameters():
        if not name.startswith("critic."):
            assert_same(parameter.grad, model.get_parameter(name).grad)
    for name in (*SHARED, "history_encoder", "actor_proprio_embedding", "actor"):
        grads = [p.grad for p in getattr(model, name).parameters()]
        assert all(g is not None and torch.isfinite(g).all() for g in grads), name
        assert sum(g.abs().sum() for g in grads) > 0, name


@pytest.mark.parametrize("training", [False, True])
def test_actor_unchanged_and_no_extra_parameters(training):
    model, obs = make_model()
    normal = ActorCriticEncoderGLAD(obs, GROUPS, 29)
    # Intentional test-only matching; production loaders reject cross-mode loading.
    state = deepcopy(model.state_dict())
    state["_extra_state"] = normal.get_extra_state()
    normal.load_state_dict(state)
    assert sum(p.numel() for p in model.parameters()) == sum(p.numel() for p in normal.parameters())
    model.train(training)
    normal.train(training)
    actor_obs = obs.exclude("clean_proprio_history", "critic")
    rng = torch.get_rng_state()
    expected = normal.act_inference(actor_obs)
    torch.set_rng_state(rng)
    assert_same(expected, model.act_inference(actor_obs))


@pytest.mark.parametrize("problem", ["missing", "alias", "length", "width", "ordinary", "ame", "lsio"])
def test_invalid_clean_configuration(problem):
    obs = clean_obs()
    groups = deepcopy(GROUPS_CLEAN)
    cls, stop = ActorCriticEncoderGLAD, True
    if problem == "missing":
        del groups["clean_history"]
    elif problem == "alias":
        groups["clean_history"] = ["proprio_history"]
    elif problem == "length":
        obs["clean_proprio_history"] = torch.randn(3, 67, 93)
    elif problem == "width":
        obs["clean_proprio_history"] = torch.randn(3, 66, 99)
    elif problem == "ordinary":
        stop = False
    elif problem == "ame":
        cls = ActorCriticEncoder
    elif problem == "lsio":
        cls = ActorCriticEncoderLSIO
    with pytest.raises(ValueError):
        cls(obs, groups, 29, critic_encoder_stop_grad=stop, critic_feature_source="actor_clean")


class CleanEnv(SyntheticEnv):
    def __init__(self):
        super().__init__()
        self.obs = clean_obs(self.num_envs)

    def step(self, actions):
        _, reward, dones, extra = super().step(actions)
        frame = torch.cat((torch.randn(self.num_envs, 64), actions), -1)
        history = torch.cat((self.obs["clean_proprio_history"][:, 1:], frame[:, None]), 1)
        history[dones] = frame[dones, None].expand(-1, 66, -1)
        self.obs["clean_proprio_history"] = history
        return self.get_observations(), reward, dones, extra


def test_runner_rollout_bootstrap_storage_save_resume(tmp_path):
    env = CleanEnv()
    runner = OnPolicyRunner(env, clean_config(), log_dir=str(tmp_path / "first"))
    model = runner.alg.policy
    before = model.history_encoder[0].weight.detach().clone()
    with patch.object(model, "_encode_terrain", wraps=model._encode_terrain) as encode:
        runner.learn(2)
    assert encode.call_count == 26  # 12 Actor calls + 12 clean values + 2 bootstraps.
    for name, buffer in model.named_buffers():
        if name.endswith("num_batches_tracked"):
            assert buffer.item() == 12
    assert env.reset_count > 0
    assert not torch.equal(before, model.history_encoder[0].weight)
    storage = runner.alg.storage
    assert storage.observations["clean_proprio_history"].shape == (4, 2, 66, 93)
    for step in range(4):
        for sample in range(2):
            identifier = step * 2 + sample
            storage.observations["policy"][step, sample, 0] = identifier
            storage.observations["clean_proprio_history"][step, sample].fill_(identifier)
    for batch, *_ in storage.mini_batch_generator(2, num_epochs=1):
        for row, identifier in enumerate(batch["policy"][:, 0]):
            assert (batch["clean_proprio_history"][row] == identifier).all()
    path = tmp_path / "new.pt"
    runner.save(str(path))
    saved = torch.load(path, weights_only=False)
    assert saved["critic_encoder_stop_grad"] is True
    assert saved["critic_feature_source"] == "actor_clean"
    assert saved["model_state_dict"]["_extra_state"]["architecture"] == "glad_lsio_clean_actor_v1"
    restored = OnPolicyRunner(CleanEnv(), clean_config(), log_dir=str(tmp_path / "resume"))
    restored.load(str(path))
    assert_same(model.state_dict(), restored.alg.policy.state_dict())
    assert_same(runner.alg.optimizer.state_dict(), restored.alg.optimizer.state_dict())
    assert restored.current_learning_iteration == runner.current_learning_iteration
    with torch.no_grad():
        assert_same(runner.get_inference_policy()(env.obs), restored.get_inference_policy()(env.obs))
    restored.load(str(path), load_optimizer=False)
    restored.learn(1)
    assert not torch.equal(model.history_encoder[0].weight, restored.alg.policy.history_encoder[0].weight)
    runner.writer.close()
    restored.writer.close()


@pytest.mark.parametrize("load_optimizer", [False, True])
def test_checkpoint_rejection_before_mutation(tmp_path, load_optimizer):
    runner = OnPolicyRunner(CleanEnv(), clean_config(), log_dir=str(tmp_path / "run"))
    runner.learn(1)
    path = tmp_path / "model.pt"
    runner.save(str(path))
    valid = torch.load(path, weights_only=False)
    before = deepcopy(runner.alg.policy.state_dict()), deepcopy(runner.alg.optimizer.state_dict()), runner.current_learning_iteration
    for mode in ("normal", "actor", "critic", "missing", "architecture", "history", "top_k"):
        bad = deepcopy(valid)
        bad["model_state_dict"]["critic.0.weight"].add_(9)
        bad["iter"] = 999
        if mode == "normal":
            bad["critic_encoder_stop_grad"] = False
        elif mode == "missing":
            del bad["critic_feature_source"]
        elif mode == "architecture":
            bad["model_state_dict"]["_extra_state"]["architecture"] = "glad_lsio_v1"
        elif mode == "history":
            bad["model_state_dict"]["_extra_state"]["long_history_length"] = 67
        elif mode == "top_k":
            bad["model_state_dict"]["_extra_state"]["top_k"] = 64
        else:
            bad["critic_feature_source"] = mode
        torch.save(bad, path)
        with pytest.raises((ValueError, RuntimeError)):
            runner.load(str(path), load_optimizer=load_optimizer)
        assert_same(before[0], runner.alg.policy.state_dict())
        assert_same(before[1], runner.alg.optimizer.state_dict())
        assert before[2] == runner.current_learning_iteration
    runner.writer.close()


def test_direct_cross_mode_loads_rejected_before_weights_change():
    model, obs = make_model()
    for stop, source in ((False, None), (True, "actor"), (True, "critic")):
        other = ActorCriticEncoderGLAD(obs, GROUPS, 29, critic_encoder_stop_grad=stop, critic_feature_source=source)
        for target, origin in ((model, other), (other, model)):
            before = deepcopy(target.state_dict())
            with pytest.raises(RuntimeError, match="architecture"):
                target.load_state_dict(origin.state_dict())
            assert_same(before, target.state_dict())


@pytest.mark.parametrize("finetune,play", [(False, False), (True, False), (False, True), (True, True)])
def test_native_clean_history_configuration(native, finetune, play):
    source = configured_source(native, finetune, play)
    old = native.layouts.make_lsio_observations(source)
    new = native.layouts.make_lsio_observations(source, include_clean_history=True)
    for key in ("policy", "proprio_history", "critic"):
        assert getattr(new, key).to_dict() == getattr(old, key).to_dict()
    h = new.clean_proprio_history
    assert h.history_length == new.proprio_history.history_length == 66
    assert not h.enable_corruption
    assert [k for k in vars(h) if k in NAMES] == list(NAMES)
    for name in NAMES:
        assert getattr(h, name).noise is None
        assert getattr(h, name).scale == getattr(new.proprio_history, name).scale
        assert getattr(h, name) is not getattr(source.critic, name)
    assert not hasattr(h, "base_lin_vel")


def test_native_clean_history_advances_with_actor_and_resets_locally(native):
    cfg = native.layouts.make_lsio_observations(configured_source(native, True), include_clean_history=True)
    widths = dict(zip(NAMES, WIDTHS), velocity_commands=3, base_lin_vel=3, height_scan=2079)
    env = SimpleNamespace(num_envs=2, device="cpu", sim=SimpleNamespace(is_playing=lambda: True),
                          values={k: torch.zeros(2, width) for k, width in widths.items()})
    for group in vars(cfg).values():
        for name, term in vars(group).items():
            if name in widths:
                term.func, term.params = read_field, {"name": name}
    manager = native.manager(cfg, env)
    manager.compute()
    for buffers in manager._group_obs_term_history_buffer.values():
        assert all(buffer._buffer is None for buffer in buffers.values())
    for step in range(70):
        for name in NAMES:
            env.values[name].fill_(step if name != "actions" else step - 1)
        obs = manager.compute(update_history=True)
    clean = obs["clean_proprio_history"]
    assert clean.shape == (2, 66, 93)
    assert_same(clean[0, :, 0], torch.arange(4, 70).float() * 0.2)
    assert_same(clean[0, :, 35], torch.arange(4, 70).float() * 0.05)
    assert_same(clean[0, :, 64], torch.arange(3, 69).float())
    assert_same(clean[:, :, 64:], obs["proprio_history"][:, :, 64:])
    assert not torch.equal(clean[:, :, :64], obs["proprio_history"][:, :, :64])
    snapshot = deepcopy(obs)
    for _ in range(3):
        fresh = manager.compute()
        for key in ("clean_proprio_history", "proprio_history"):
            assert_same(snapshot[key], fresh[key])
    manager.reset(env_ids=torch.tensor([0]))
    for name in NAMES:
        env.values[name][0] = 0 if name == "actions" else -10
        env.values[name][1] = 70 if name != "actions" else 69
    reset = manager.compute(update_history=True)
    for key in ("clean_proprio_history", "proprio_history"):
        assert_same(reset[key][0], reset[key][0, -1:].expand(66, -1))
        assert_same(reset[key][1, :-1], snapshot[key][1, 1:])
        assert (reset[key][0, :, 64:] == 0).all()


def test_task_registration_and_native_runner_config(native):
    tree = ast.parse((TASK / "29dof/__init__.py").read_text())
    tree.body = [n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
    registrations = {}
    exec(compile(tree, "registrations", "exec"), {
        "__name__": "tasks", "agents": SimpleNamespace(__name__="agents"),
        "gym": SimpleNamespace(register=lambda **kw: registrations.update({kw["id"]: kw})),
    })
    for play in (False, True):
        item = registrations[f"AME-G1-29DOF-GLAD-CriticCleanActorStopGrad{'-Play' if play else ''}-v0"]
        assert item["kwargs"]["env_cfg_entry_point"].endswith(f":G1LSIOCleanActorEnvCfg{'_PLAY' if play else ''}")
        assert item["kwargs"]["rsl_rl_cfg_entry_point"].endswith(":G1AMEGLADCriticCleanActorStopGradPPORunnerCfg")
    isaaclab_rl = pytest.importorskip("isaaclab_rl")
    package = ModuleType("_clean_actor_rl_cfg")
    package.__path__ = [str(Path(isaaclab_rl.__file__).parent / "rsl_rl")]
    with patch.dict(sys.modules, {package.__name__: package}):
        cfg_module = importlib.import_module(f"{package.__name__}.rl_cfg")
        with patch.dict(sys.modules, {"isaaclab_rl.rsl_rl": cfg_module}):
            agents = load_file("_clean_actor_agent_cfg", TASK / "agents/ame_rsl_rl_ppo_cfg.py")
    cfg = agents.G1AMEGLADCriticCleanActorStopGradPPORunnerCfg()
    expected = agents.G1AMEGLADPPORunnerCfg().to_dict()
    expected["experiment_name"] = "g1_ame_glad_critic_clean_actor_stop_grad"
    expected["policy"].update(critic_encoder_stop_grad=True, critic_feature_source="actor_clean")
    expected["obs_groups"] = GROUPS_CLEAN
    assert cfg.to_dict() == expected
    assert cfg.resume is False
