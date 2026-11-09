import csv

import numpy as np
import torch

from mountaincar.agent import Agent, clip_action
from mountaincar.evaluation import evaluate_policy
from mountaincar.train import compute_gae, train


def test_gae_termination_truncation_and_fragment_boundary():
    # Three independent environments: terminal, time limit, and continuing.
    rewards = np.array([[1, 1, 1], [2, 2, 2]], dtype=np.float32)
    values = np.array([[0.5, 0.5, 0.5], [1, 1, 1]], dtype=np.float32)
    next_values = np.array([[1, 1, 1], [3, 3, 3]], dtype=np.float32)
    terminated = np.array([[True, False, False], [False, False, False]])
    truncated = np.array([[False, True, False], [False, False, False]])
    advantages, returns = compute_gae(rewards, values, next_values, terminated, truncated, gamma=0.9, gae_lambda=0.8)
    # Last step bootstraps 3 but has no later trace: 2 + .9*3 - 1 = 3.7.
    # First deltas: .5 at termination, 1.4 at truncation/continuation.
    expected = np.array([[0.5, 1.4, 1.4 + .9 * .8 * 3.7], [3.7, 3.7, 3.7]])
    np.testing.assert_allclose(advantages, expected, rtol=1e-6)
    np.testing.assert_allclose(returns, expected + values, rtol=1e-6)


def test_raw_action_ratio_and_clipping():
    agent = Agent()
    obs = torch.tensor([[0.0, 0.01], [-0.5, -0.03]])
    action, old_logprob, _ = agent.act(obs)
    new_logprob = agent.distribution(obs).log_prob(action).sum(-1)
    torch.testing.assert_close((new_logprob - old_logprob).exp(), torch.ones(2))
    np.testing.assert_array_equal(clip_action(np.array([-2., -.25, 3.])), [-1., -.25, 1.])


def test_checkpoint_preserves_deterministic_action(tmp_path):
    torch.manual_seed(4)
    agent = Agent()
    observations = torch.tensor([[-0.5, 0.0], [0.1, 0.02]])
    before = agent.act(observations, deterministic=True)
    path = tmp_path / 'agent.pt'
    agent.save(path, {'purpose': 'test'})
    after = Agent.load(path).act(observations, deterministic=True)
    for left, right in zip(before, after):
        torch.testing.assert_close(left, right)


def test_tiny_train_and_evaluate(tmp_path):
    config = {'num_envs': 2, 'rollout_steps': 4, 'total_timesteps': 16,
              'minibatch_size': 4, 'update_epochs': 1, 'eval_every_updates': 100,
              'torch_threads': 1}
    result = train(config, 7, tmp_path)
    assert result['steps'] == 16
    with (tmp_path / 'training.csv').open(newline='', encoding='utf-8') as file:
        rows = list(csv.DictReader(file))
    assert rows
    assert all(np.isfinite(float(value)) for row in rows for value in row.values())
    evaluation = evaluate_policy(Agent.load(tmp_path / 'best.pt'), [20000])
    assert len(evaluation['episodes']) == 1
    assert all(np.isfinite(value) for value in evaluation['summary'].values())
