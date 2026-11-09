"""Evaluate policies on fixed seeds, counting raw environment steps."""
import gymnasium as gym
import numpy as np
import torch
from .controls import step_action

VALIDATION_SEEDS = tuple(range(10000, 10020))
TEST_SEEDS = tuple(range(20000, 20100))


def action_repeat(policy):
    return 20 if policy == "random_hold" else 1 if isinstance(policy, str) else getattr(policy, "action_repeat", 1)


def choose_action(policy, observation, rng, deterministic=True):
    if isinstance(policy, str):
        if policy in ("random", "random_hold"):
            return np.array([rng.uniform(-1, 1)], dtype=np.float32)
        if policy == "heuristic":
            return np.array([1.0 if observation[1] >= 0 else -1.0], dtype=np.float32)
        raise ValueError(f"Unknown baseline: {policy}")
    with torch.no_grad():
        action, _, _ = policy.act(torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0), deterministic=deterministic)
    return np.clip(action.detach().cpu().numpy().reshape(-1), -1, 1)


def evaluate_policy(policy, seeds, deterministic=True):
    seeds = list(seeds)
    if not seeds:
        raise ValueError("At least one evaluation seed is required")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(12345)
        episodes = _batched_episodes(policy, seeds, deterministic)
    fields = {"mean_reward": "reward", "success_rate": "success", "mean_length": "length", "mean_action_energy": "action_energy", "mean_max_position": "max_position", "mean_max_speed": "max_speed"}
    return {"summary": {key: float(np.mean([row[value] for row in episodes])) for key, value in fields.items()}, "episodes": episodes}


def _batched_episodes(policy, seeds, deterministic=True):
    episodes = []
    repeat = action_repeat(policy)
    for start in range(0, len(seeds), 20):
        batch = seeds[start:start + 20]
        envs = []
        try:
            for _ in batch:
                envs.append(gym.make("MountainCarContinuous-v0"))
            observations = [env.reset(seed=int(seed))[0] for env, seed in zip(envs, batch)]
            rngs = [np.random.default_rng(seed) for seed in batch]
            rows = [{"seed": int(seed), "reward": 0.0, "success": False, "length": 0, "action_energy": 0.0, "max_position": float(obs[0]), "max_speed": abs(float(obs[1]))} for seed, obs in zip(batch, observations)]
            active = list(range(len(batch)))
            while active:
                if isinstance(policy, str):
                    actions = [choose_action(policy, observations[i], rngs[i]) for i in active]
                else:
                    with torch.no_grad():
                        actions, _, _ = policy.act(torch.as_tensor(np.stack([observations[i] for i in active]), dtype=torch.float32), deterministic=deterministic)
                    actions = np.clip(actions.detach().cpu().numpy(), -1, 1)
                remaining = []
                for i, action in zip(active, actions):
                    observations[i], _, terminated, truncated, info = step_action(envs[i], action, repeat, gamma=1.0, max_steps=999 - rows[i]["length"])
                    row = rows[i]
                    row["reward"] += info["raw_reward"]
                    row["action_energy"] += info["action_energy"]
                    row["length"] += info["duration"]
                    row["max_position"] = max(row["max_position"], info["max_position"])
                    row["max_speed"] = max(row["max_speed"], info["max_speed"])
                    row["success"] = bool(terminated)
                    if not (terminated or truncated) and row["length"] < 999:
                        remaining.append(i)
                active = remaining
            episodes.extend(rows)
        finally:
            for env in envs:
                env.close()
    return episodes
