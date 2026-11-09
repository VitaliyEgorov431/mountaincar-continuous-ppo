import numpy as np
import torch
import pytest

from mountaincar.agent import Agent
from mountaincar.controls import step_action
from mountaincar.train import compute_gae, train


class FakeEnv:
    def __init__(self, end=99, truncated=False):
        self.i = 0
        self.end = end
        self.truncated = truncated
    def step(self, action):
        assert np.all(np.abs(action) <= 1)
        self.i += 1
        done = self.i == self.end
        return np.array([self.i, -self.i / 10]), float(self.i), done and not self.truncated, done and self.truncated, {}


@pytest.mark.parametrize('truncated', [False, True])
def test_step_action_breaks_on_done(truncated):
    obs, reward, term, trunc, info = step_action(FakeEnv(2, truncated), [2], 20, .5)
    assert reward == 2
    assert (term, trunc) == (not truncated, truncated)
    assert info == {'duration': 2, 'raw_reward': 3, 'action_energy': 2, 'max_position': 2, 'max_speed': .2}
    assert obs[0] == 2


def test_step_action_budget_cap():
    _, reward, term, trunc, info = step_action(FakeEnv(), [0], 20, max_steps=3)
    assert (reward, term, trunc, info['duration']) == (6, False, False, 3)


def test_variable_duration_gae():
    adv, returns = compute_gae(np.array([[1.], [2.]]), np.array([[.5], [.6]]), np.array([[.6], [.7]]), np.array([[0], [0]]), np.array([[0], [1]]), .9, .8, np.array([[2], [3]]))
    delta2 = 2 + .9**3 * .7 - .6
    assert adv[:, 0] == pytest.approx([1 + .9**2 * .6 - .5 + .9**2 * .8 * delta2, delta2])
    assert returns[:, 0] == pytest.approx(adv[:, 0] + [.5, .6])


def test_checkpoint_repeat_and_legacy(tmp_path):
    agent = Agent()
    agent.action_repeat = 20
    agent.save(tmp_path / 'new.pt')
    assert Agent.load(tmp_path / 'new.pt').action_repeat == 20
    torch.save({'state_dict': agent.state_dict(), 'metadata': {}}, tmp_path / 'old.pt')
    assert Agent.load(tmp_path / 'old.pt').action_repeat == 1


def test_exact_budget_smaller_than_env_count(tmp_path, monkeypatch):
    monkeypatch.setattr('mountaincar.train.VALIDATION_SEEDS', [123])
    summary = train({'total_timesteps': 7, 'num_envs': 8, 'update_epochs': 1, 'rollout_steps': 1}, 7, tmp_path)
    assert summary['steps'] == 7
    assert summary['decisions'] == 7


def test_uneven_budget_with_early_termination(tmp_path, monkeypatch):
    class EarlyTerminationEnv:
        total_steps = 0
        def __init__(self):
            self.episode_steps = 0
        def reset(self, seed=None):
            self.episode_steps = 0
            return np.array([-0.5, 0.0], dtype=np.float32), {}
        def step(self, action):
            type(self).total_steps += 1
            self.episode_steps += 1
            return np.array([-0.5, 0.0], dtype=np.float32), 1.0, self.episode_steps == 3, False, {}
        def close(self):
            pass

    monkeypatch.setattr('mountaincar.train.gym.make', lambda *args, **kwargs: EarlyTerminationEnv())
    monkeypatch.setattr('mountaincar.train.evaluate_policy', lambda *args, **kwargs: {'summary': {'mean_reward': 0.0, 'success_rate': 0.0, 'mean_length': 3.0, 'mean_action_energy': 0.0}})
    summary = train({'total_timesteps': 403, 'num_envs': 8, 'rollout_steps': 4, 'update_epochs': 1, 'minibatch_size': 64}, 7, tmp_path)
    assert summary['steps'] == 403
    assert EarlyTerminationEnv.total_steps == 403
