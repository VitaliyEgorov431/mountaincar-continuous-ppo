"""Synchronous PPO training with time-limit-aware GAE."""

import csv
import json
import random
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from .agent import Agent
from .controls import step_action
from .evaluation import evaluate_policy


DEFAULT_CONFIG = {
    "num_envs": 8, "rollout_steps": 32, "total_timesteps": 50_000, "action_repeat": 20,
    "minibatch_size": 64, "update_epochs": 10, "learning_rate": 3e-4,
    "gamma": 0.99, "gae_lambda": 0.95, "clip_coef": 0.2,
    "value_coef": 0.5, "entropy_coef": 0.01, "max_grad_norm": 0.5,
    "eval_every_updates": 2, "torch_threads": 1,
}
VALIDATION_SEEDS = list(range(10000, 10020))


def compute_gae(rewards, values, next_values, terminated, truncated, gamma=0.99, gae_lambda=0.95, durations=None):
    rewards = np.asarray(rewards)
    durations = np.ones_like(rewards) if durations is None else np.asarray(durations)
    advantages = np.zeros_like(rewards, dtype=np.float32)
    carry = np.zeros(rewards.shape[1], dtype=np.float32)
    for t in range(len(rewards) - 1, -1, -1):
        delta = rewards[t] + gamma ** durations[t] * next_values[t] * (1 - terminated[t]) - values[t]
        carry = delta + gamma ** durations[t] * gae_lambda * (1 - np.logical_or(terminated[t], truncated[t])) * carry
        advantages[t] = carry
    return advantages, advantages + values


def _writer(path, fields):
    file = path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(file, fieldnames=fields)
    writer.writeheader()
    return file, writer


def train(config, seed, output_dir):
    started = time.monotonic()
    cfg = {**DEFAULT_CONFIG, **config}
    for key in ("num_envs", "rollout_steps", "total_timesteps", "minibatch_size", "update_epochs", "eval_every_updates", "torch_threads", "action_repeat"):
        if not isinstance(cfg[key], int) or cfg[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "run_summary.json").unlink(missing_ok=True)
    (output / "config.json").write_text(json.dumps({**cfg, "seed": seed}, indent=2), encoding="utf-8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(cfg["torch_threads"])
    agent = Agent()
    agent.action_repeat = cfg["action_repeat"]
    # Fuse optimizer operations to reduce overhead while retaining Adam's equations.
    optimizer = torch.optim.Adam(agent.parameters(), lr=cfg["learning_rate"], eps=1e-5, fused=True)
    envs = [gym.make("MountainCarContinuous-v0") for _ in range(cfg["num_envs"])]
    files = []
    try:
        obs = np.stack([env.reset(seed=seed + i)[0] for i, env in enumerate(envs)])
        episode_returns = np.zeros(len(envs))
        episode_lengths = np.zeros(len(envs), dtype=int)
        episode_max_positions = obs[:, 0].copy()
        episode_max_speeds = np.abs(obs[:, 1]).copy()
        episode_energies = np.zeros(len(envs))
        decisions = training_successes = 0
        f, metrics = _writer(output / "training.csv", ["step", "update", "policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction", "policy_std", "training_successes", "decisions"])
        files.append(f)
        f, episodes = _writer(output / "episodes.csv", ["step", "env", "return", "length", "success", "max_position", "max_speed", "action_energy"])
        files.append(f)
        validation_fields = ["step", "update", "mean_reward", "success_rate", "mean_length", "mean_action_energy"]
        f, validation = _writer(output / "validation.csv", validation_fields)
        files.append(f)
        f, stochastic_validation = _writer(output / "stochastic_validation.csv", validation_fields)
        files.append(f)
        steps = update = 0
        best = None
        max_validation_success_rate = 0.0
        while steps < cfg["total_timesteps"]:
            n = min(len(envs), cfg["total_timesteps"] - steps)
            horizon = cfg["rollout_steps"]
            rollout = {key: [] for key in ("obs", "actions", "log_probs", "values", "rewards", "next_values", "terminated", "truncated", "durations")}
            for _ in range(horizon):
                remaining = cfg["total_timesteps"] - steps
                if remaining < n:
                    break
                cap = max(1, min(cfg["action_repeat"], remaining // n))
                current = torch.as_tensor(obs[:n], dtype=torch.float32)
                with torch.no_grad():
                    actions, log_probs, values = agent.act(current)
                next_obs, rewards, terminated, truncated, durations = [], [], [], [], []
                for i in range(n):
                    nxt, reward, term, trunc, info = step_action(envs[i], actions[i].numpy(), cfg["action_repeat"], cfg["gamma"], max_steps=cap)
                    next_obs.append(nxt)
                    rewards.append(reward)
                    terminated.append(term)
                    truncated.append(trunc)
                    durations.append(info["duration"])
                    episode_returns[i] += info["raw_reward"]
                    episode_lengths[i] += info["duration"]
                    episode_max_positions[i] = max(episode_max_positions[i], info["max_position"])
                    episode_max_speeds[i] = max(episode_max_speeds[i], info["max_speed"])
                    episode_energies[i] += info["action_energy"]
                with torch.no_grad():
                    next_values = agent.value(torch.as_tensor(np.stack(next_obs), dtype=torch.float32)).numpy()
                for key, data in (("obs", obs[:n].copy()), ("actions", actions.numpy()), ("log_probs", log_probs.numpy()), ("values", values.numpy()), ("rewards", rewards), ("next_values", next_values), ("terminated", terminated), ("truncated", truncated), ("durations", durations)):
                    rollout[key].append(data)
                for i in range(n):
                    if terminated[i] or truncated[i]:
                        training_successes += int(terminated[i])
                        episodes.writerow({"step": steps + sum(durations[:i + 1]), "env": i, "return": episode_returns[i], "length": episode_lengths[i], "success": bool(terminated[i]), "max_position": episode_max_positions[i], "max_speed": episode_max_speeds[i], "action_energy": episode_energies[i]})
                        next_obs[i] = envs[i].reset()[0]
                        episode_returns[i] = 0
                        episode_lengths[i] = 0
                        episode_max_positions[i] = next_obs[i][0]
                        episode_max_speeds[i] = abs(next_obs[i][1])
                        episode_energies[i] = 0
                    obs[i] = next_obs[i]
                steps += sum(durations)
                decisions += n
            data = {key: np.asarray(value) for key, value in rollout.items()}
            advantages, returns = compute_gae(data["rewards"], data["values"], data["next_values"], data["terminated"], data["truncated"], cfg["gamma"], cfg["gae_lambda"], data["durations"])
            batch_obs = torch.tensor(data["obs"].reshape(-1, 2), dtype=torch.float32)
            batch_actions = torch.tensor(data["actions"].reshape(-1, 1), dtype=torch.float32)
            old_log_probs = torch.tensor(data["log_probs"].flatten(), dtype=torch.float32)
            batch_advantages = torch.tensor(advantages.flatten(), dtype=torch.float32)
            batch_returns = torch.tensor(returns.flatten(), dtype=torch.float32)
            batch_advantages = (batch_advantages - batch_advantages.mean()) / (batch_advantages.std(unbiased=False) + 1e-8)
            stats = []
            for _ in range(cfg["update_epochs"]):
                indices = np.random.permutation(len(batch_obs))
                for start in range(0, len(indices), cfg["minibatch_size"]):
                    idx = indices[start:start + cfg["minibatch_size"]]
                    distribution = agent.distribution(batch_obs[idx])
                    log_probs = distribution.log_prob(batch_actions[idx]).sum(-1)
                    log_ratio = log_probs - old_log_probs[idx]
                    ratio = log_ratio.exp()
                    policy_loss = -torch.min(ratio * batch_advantages[idx], ratio.clamp(1 - cfg["clip_coef"], 1 + cfg["clip_coef"]) * batch_advantages[idx]).mean()
                    value_loss = ((agent.value(batch_obs[idx]) - batch_returns[idx]) ** 2).mean()
                    entropy = distribution.entropy().sum(-1).mean()
                    loss = policy_loss + cfg["value_coef"] * value_loss - cfg["entropy_coef"] * entropy
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Non-finite PPO loss")
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(agent.parameters(), cfg["max_grad_norm"], error_if_nonfinite=True)
                    optimizer.step()
                    with torch.no_grad():
                        stats.append([policy_loss.item(), value_loss.item(), entropy.item(), ((ratio - 1) - log_ratio).mean().item(), ((ratio - 1).abs() > cfg["clip_coef"]).float().mean().item()])
            update += 1
            means = np.mean(stats, axis=0)
            metrics.writerow(dict(zip(["step", "update", "policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction", "policy_std", "training_successes", "decisions"], [steps, update, *means, agent.log_std.exp().item(), training_successes, decisions])))
            if update % cfg["eval_every_updates"] == 0 or steps == cfg["total_timesteps"]:
                result = evaluate_policy(agent, VALIDATION_SEEDS)["summary"]
                validation.writerow({key: ({"step": steps, "update": update, **result})[key] for key in validation_fields})
                stochastic = evaluate_policy(agent, VALIDATION_SEEDS, deterministic=False)["summary"]
                stochastic_validation.writerow({key: ({"step": steps, "update": update, **stochastic})[key] for key in validation_fields})
                if not all(np.isfinite(result[key]) for key in validation_fields[2:]):
                    raise FloatingPointError("Non-finite validation metric")
                max_validation_success_rate = max(max_validation_success_rate, result["success_rate"])
                score = (result["mean_reward"], result["success_rate"])
                if best is None or score > best[0]:
                    best = (score, steps, dict(result))
                    agent.save(output / "best.pt", {"config": cfg, "seed": seed, "steps": steps,
                               "elapsed_seconds": time.monotonic() - started,
                               "gymnasium_version": gym.__version__, "torch_version": str(torch.__version__),
                               "numpy_version": np.__version__})
                print(f"seed={seed} steps={steps} mean_reward={score[0]:.3f} success_rate={score[1]:.3f}", flush=True)
            for file in files:
                file.flush()
        agent.save(output / "final.pt", {"config": cfg, "seed": seed, "steps": steps,
                   "elapsed_seconds": time.monotonic() - started,
                   "gymnasium_version": gym.__version__, "torch_version": str(torch.__version__),
                   "numpy_version": np.__version__})
        summary = {"config": cfg, "seed": seed, "steps": steps, "decisions": decisions, "training_successes": training_successes, "best_step": best[1],
                   "best_mean_reward": best[0][0], "best_success_rate": best[0][1],
                   "best_checkpoint": str(output / "best.pt"), "best_validation": best[2],
                   "max_validation_success_rate": max_validation_success_rate,
                   "elapsed_seconds": time.monotonic() - started,
                   "gymnasium_version": gym.__version__, "torch_version": str(torch.__version__),
                   "numpy_version": np.__version__}
        (output / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary
    finally:
        for env in envs:
            env.close()
        for file in files:
            file.close()
