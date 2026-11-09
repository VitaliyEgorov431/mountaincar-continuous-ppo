import numpy as np
import pytest
import torch
import csv
import json
import gymnasium as gym
from mountaincar.agent import Agent
from mountaincar.evaluation import evaluate_policy
from mountaincar.experiments import _passed, run_experiment
from mountaincar.train import DEFAULT_CONFIG


def test_repeat_metrics_and_old_checkpoint(tmp_path):
    torch.set_num_threads(1)
    agent = Agent()
    agent.save(tmp_path / "old.pt")
    old = Agent.load(tmp_path / "old.pt")
    assert old.action_repeat == 1
    old.action_repeat = 20
    result = evaluate_policy(old, [10000])["episodes"][0]
    assert 1 <= result["length"] <= 999
    assert result["action_energy"] >= 0
    assert result["max_position"] >= -0.6
    assert result["max_speed"] >= 0
    if not result["success"]:
        assert result["reward"] == pytest.approx(-0.1 * result["action_energy"], abs=1e-5)


def test_stochastic_evaluation_does_not_change_training_rng():
    torch.set_num_threads(1)
    agent = Agent()
    agent.action_repeat = 20
    state = torch.random.get_rng_state().clone()
    first = evaluate_policy(agent, [10000], deterministic=False)
    assert torch.equal(state, torch.random.get_rng_state())
    assert first == evaluate_policy(agent, [10000], deterministic=False)


def test_random_baselines_repeat_and_seeded():
    first = evaluate_policy("random_hold", [10000])
    assert first == evaluate_policy("random_hold", [10000])
    assert first != evaluate_policy("random", [10000])


def test_held_action_metrics_match_independent_raw_steps():
    class ConstantAgent:
        action_repeat = 20
        def act(self, obs, deterministic=True):
            return torch.full((len(obs), 1), 0.25), None, None
    actual = evaluate_policy(ConstantAgent(), [10000])["episodes"][0]
    env = gym.make("MountainCarContinuous-v0")
    try:
        obs, _ = env.reset(seed=10000)
        reward = energy = 0.0
        position, speed = float(obs[0]), abs(float(obs[1]))
        for length in range(1, 1000):
            obs, r, terminated, truncated, _ = env.step(np.array([0.25], dtype=np.float32))
            reward += r
            energy += 0.25 ** 2
            position = max(position, float(obs[0]))
            speed = max(speed, abs(float(obs[1])))
            if terminated or truncated:
                break
    finally:
        env.close()
    assert actual["length"] == length
    assert actual["success"] == terminated
    assert actual["reward"] == pytest.approx(reward)
    assert actual["action_energy"] == pytest.approx(energy)
    assert actual["max_position"] == pytest.approx(position)
    assert actual["max_speed"] == pytest.approx(speed)


def test_gate_requires_three_consecutive_checks_and_matching_config(tmp_path):
    config = {**DEFAULT_CONFIG, "total_timesteps": 200000}
    (tmp_path / "best.pt").touch()
    (tmp_path / "run_summary.json").write_text(json.dumps({"seed": 0, "config": config, "steps": 200000}), encoding="utf-8")
    path = tmp_path / "validation.csv"
    def write_rates(rates):
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["success_rate"])
            writer.writeheader()
            writer.writerows({"success_rate": rate} for rate in rates)
    write_rates([0.9, 0.8, 0.9, 0.9])
    assert not _passed(tmp_path, config, 0)
    write_rates([0.8, 0.9, 1.0, 0.95])
    assert _passed(tmp_path, config, 0)
    assert not _passed(tmp_path, {**config, "entropy_coef": 0.123}, 0)


@pytest.mark.parametrize("phase", ["confirm", "final", "report"])
def test_missing_pilot_blocks_later_phases(tmp_path, monkeypatch, phase):
    def unexpected(*args, **kwargs):
        pytest.fail("Training or testing ran before the gate")
    monkeypatch.setattr("mountaincar.experiments.train", unexpected)
    monkeypatch.setattr("mountaincar.experiments.evaluate_policy", unexpected)
    with pytest.raises(ValueError, match="Pilot"):
        run_experiment({"total_timesteps": 200000}, tmp_path, phase=phase)


@pytest.mark.parametrize("phase", ["final", "report"])
def test_missing_confirmation_blocks_final_phases(tmp_path, monkeypatch, phase):
    def unexpected(*args, **kwargs):
        pytest.fail("Training or testing ran before the gate")
    monkeypatch.setattr("mountaincar.experiments.train", unexpected)
    monkeypatch.setattr("mountaincar.experiments.evaluate_policy", unexpected)
    monkeypatch.setattr("mountaincar.experiments._passed", lambda directory, config, seed: seed == 0)
    with pytest.raises(ValueError, match="confirmation"):
        run_experiment({"total_timesteps": 200000}, tmp_path, phase=phase)
