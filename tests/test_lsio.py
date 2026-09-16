"""CPU tests: run with PYTHONPATH=rsl_rl python -m pytest tests/test_lsio.py."""

from copy import deepcopy
from pathlib import Path

import pytest
import torch
from tensordict import TensorDict

from rsl_rl.modules import ActorCriticEncoder, ActorCriticEncoderLSIO
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.storage import RolloutStorage


GROUPS = {"policy": ["policy"], "critic": ["critic"], "history": ["proprio_history"]}
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.manual_seed(42)
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def observations(batch=3, history=66):
    return TensorDict({
        "policy": torch.randn(batch, 2082),
        "critic": torch.randn(batch, 2178),
        "proprio_history": torch.randn(batch, history, 93),
    }, batch_size=[batch])


@pytest.mark.parametrize("attach_global", [False, True])
def test_dimensions_actor_critic_and_gradients(attach_global):
    obs = observations()
    model = ActorCriticEncoderLSIO(obs, GROUPS, 29, attach_global=attach_global)
    assert model.is_recurrent is False
    assert model.history_encoder(obs["proprio_history"].transpose(1, 2)).shape == (3, 144)
    assert model.actor_proprio_dim == 519
    assert model.actor[0].in_features == (647 if attach_global else 583)
    assert model.critic_proprio_dim == 99
    assert model.critic[0].in_features == (227 if attach_global else 163)
    packed = model.get_actor_obs(obs)
    torch.testing.assert_close(packed[:, 144:516], obs["proprio_history"][:, -4:].flatten(1))
    torch.testing.assert_close(packed[:, 516:519], obs["policy"][:, :3])
    actions, attention = model.act_inference(obs)
    assert actions.shape == (3, 29)
    assert attention.shape == (3, 1, 187)
    torch.testing.assert_close(attention.sum(-1), torch.ones(3, 1))
    assert model.evaluate(obs).shape == (3, 1)
    (actions.square().mean() + model.evaluate(obs).square().mean()).backward()
    for name, parameter in model.named_parameters():
        if name == "std":  # Deterministic inference does not use distribution scale.
            continue
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    assert model.history_encoder[0].weight.grad.abs().sum() > 0
    assert model.actor_proprio_embedding.weight.grad.abs().sum() > 0
    assert model.critic_proprio_embedding.weight.grad.abs().sum() > 0


def test_both_history_paths_and_critic_independence():
    obs = observations()
    model = ActorCriticEncoderLSIO(obs, GROUPS, 29).eval()
    snapshot = obs.clone()
    with torch.no_grad():
        baseline, weights = model.act_inference(obs)
        value = model.evaluate(obs)
        early = obs.clone()
        early["proprio_history"][:, :20] += 5
        assert not torch.allclose(baseline, model.act_inference(early)[0])
        torch.testing.assert_close(value, model.evaluate(early), rtol=0, atol=0)
        changed_current = obs.clone()
        changed_current["policy"] += 1
        torch.testing.assert_close(value, model.evaluate(changed_current), rtol=0, atol=0)
        # Isolate the direct short-history bypass by disabling the long encoder.
        for parameter in model.history_encoder.parameters():
            parameter.zero_()
        baseline = model.act_inference(obs)[0]
        torch.testing.assert_close(baseline, model.act_inference(early)[0], rtol=0, atol=0)
        recent = obs.clone()
        recent["proprio_history"][:, -4:] += 5
        assert not torch.allclose(baseline, model.act_inference(recent)[0])
        model.act(obs)
        model.act_inference(obs)
        model.evaluate(obs)
    for key in obs.keys():
        torch.testing.assert_close(obs[key], snapshot[key], rtol=0, atol=0)


@pytest.mark.parametrize("kwargs", [
    {"long_history_length": 67}, {"short_history_length": 67},
    {"long_history_length": 14}, {"actor_obs_normalization": True},
    {"critic_obs_normalization": True},
])
def test_reject_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        ActorCriticEncoderLSIO(observations(), GROUPS, 29, **kwargs)


def test_reject_invalid_observation_layout():
    obs = observations()
    with pytest.raises(ValueError, match="obs_groups"):
        ActorCriticEncoderLSIO(obs, {"policy": ["policy"], "critic": ["critic"]}, 29)
    obs["policy"] = torch.zeros(3, 2175)
    with pytest.raises(ValueError, match="3 velocity commands"):
        ActorCriticEncoderLSIO(obs, GROUPS, 29)
    obs = observations()
    model = ActorCriticEncoderLSIO(obs, GROUPS, 29)
    obs["proprio_history"] = torch.zeros(3, 93, 66)
    with pytest.raises(ValueError, match="history must have shape"):
        model.act_inference(obs)


def test_storage_preserves_snapshots_and_shuffles_whole_histories():
    obs = observations(batch=2)
    storage = RolloutStorage("rl", 2, 4, obs, [29])
    expected = {}
    pattern = torch.arange(66 * 93).reshape(66, 93).float() / 10000
    for step in range(4):
        ids = torch.arange(2) + step * 2
        obs["policy"][:, 0] = ids
        obs["proprio_history"][:] = ids[:, None, None] + pattern
        for row, sample_id in enumerate(ids.tolist()):
            expected[sample_id] = obs["proprio_history"][row].clone()
        transition = RolloutStorage.Transition()
        transition.observations = obs
        transition.actions = ids[:, None].expand(-1, 29)
        transition.rewards = torch.zeros(2)
        transition.dones = torch.zeros(2)
        transition.values = torch.zeros(2, 1)
        transition.actions_log_prob = torch.zeros(2)
        transition.action_mean = torch.zeros(2, 29)
        transition.action_sigma = torch.ones(2, 29)
        storage.add_transitions(transition)
    obs["proprio_history"].fill_(-999)
    seen = []
    for batch, actions, *_ in storage.mini_batch_generator(2, num_epochs=2):
        assert batch["proprio_history"].shape == (4, 66, 93)
        for row, sample_id in enumerate(batch["policy"][:, 0].int().tolist()):
            torch.testing.assert_close(batch["proprio_history"][row], expected[sample_id], rtol=0, atol=0)
            assert actions[row, 0] == sample_id
            seen.append(sample_id)
    assert sorted(seen) == sorted(list(range(8)) * 2)


class SyntheticEnv:
    """Small deterministic vector environment; no physics or simulator claims."""

    num_envs, num_actions, device = 2, 29, "cpu"
    max_episode_length = 3

    def __init__(self):
        self.obs = observations(self.num_envs)
        self.episode_length_buf = torch.tensor([0, 1])
        self.reset_count = 0

    def get_observations(self):
        return self.obs.clone()

    def step(self, actions):
        self.episode_length_buf += 1
        dones = self.episode_length_buf >= self.max_episode_length
        self.obs = self.obs.clone()
        frame = torch.cat((torch.randn(self.num_envs, 64), actions), dim=-1)
        self.obs["proprio_history"] = torch.cat((self.obs["proprio_history"][:, 1:], frame[:, None]), dim=1)
        if dones.any():
            self.reset_count += int(dones.sum())
            self.episode_length_buf[dones] = 0
            self.obs["proprio_history"][dones] = 0
        return self.get_observations(), -actions.square().mean(-1), dones, {}


def runner_config():
    return {
        "num_steps_per_env": 4, "save_interval": 1, "obs_groups": deepcopy(GROUPS),
        "policy": {"class_name": "ActorCriticEncoderLSIO"},
        "algorithm": {"class_name": "PPO", "num_learning_epochs": 1, "num_mini_batches": 2},
    }


def test_cpu_ppo_runner_save_restore_and_resume(tmp_path):
    env = SyntheticEnv()
    runner = OnPolicyRunner(env, runner_config(), log_dir=str(tmp_path / "first"))
    before = runner.alg.policy.history_encoder[0].weight.detach().clone()
    runner.learn(2)
    assert env.reset_count > 0
    assert not torch.equal(before, runner.alg.policy.history_encoder[0].weight)
    assert all(torch.isfinite(p).all() for p in runner.alg.policy.parameters())
    checkpoint = tmp_path / "lsio.pt"
    runner.save(str(checkpoint))
    obs = env.get_observations()
    with torch.no_grad():
        expected = runner.get_inference_policy()(obs)
    restored = OnPolicyRunner(SyntheticEnv(), runner_config(), log_dir=str(tmp_path / "resume"))
    restored.load(str(checkpoint), map_location="cpu")
    assert restored.current_learning_iteration == runner.current_learning_iteration
    assert restored.alg.optimizer.state_dict()["state"]
    for state in runner.alg.optimizer.state_dict()["state"]:
        for key, value in runner.alg.optimizer.state_dict()["state"][state].items():
            torch.testing.assert_close(restored.alg.optimizer.state_dict()["state"][state][key], value)
    with torch.no_grad():
        actual = restored.get_inference_policy()(obs)
    for left, right in zip(expected, actual):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    restored.learn(1)
    assert all(torch.isfinite(p).all() for p in restored.alg.policy.parameters())
    runner.writer.close()
    restored.writer.close()


@pytest.mark.parametrize("name,attach_global", [("ame1.pt", False), ("ame2.pt", True)])
def test_real_legacy_checkpoints(name, attach_global):
    checkpoint = ROOT / "pretrained" / name
    if not checkpoint.exists():
        pytest.skip(f"Not present: {checkpoint}")
    state = torch.load(checkpoint, weights_only=False, map_location="cpu")["model_state_dict"]
    obs = TensorDict({"policy": torch.randn(2, 2175), "critic": torch.randn(2, 2178)}, [2])
    model = ActorCriticEncoder(obs, GROUPS, 29, attach_global=attach_global).eval()
    assert model.load_state_dict(state)
    with torch.no_grad():
        actions, attention = model.act_inference(obs)
        assert torch.isfinite(actions).all()
        assert attention.shape == (2, 1, 187)
        assert torch.isfinite(model.evaluate(obs)).all()
    lsio = ActorCriticEncoderLSIO(observations(), GROUPS, 29)
    with pytest.raises(RuntimeError, match="legacy AME"):
        lsio.load_state_dict(state)


def test_checkpoint_rejects_same_size_but_different_history():
    model = ActorCriticEncoderLSIO(observations(), GROUPS, 29)
    changed = ActorCriticEncoderLSIO(observations(history=67), GROUPS, 29, long_history_length=67)
    assert model.long_latent_dim == changed.long_latent_dim
    with pytest.raises(RuntimeError, match="architecture/history"):
        changed.load_state_dict(model.state_dict())
