"""Execute a bounded sequence of raw environment steps."""
import numpy as np


def step_action(env, action, repeat, gamma=1.0, max_steps=None):
    if repeat <= 0 or (max_steps is not None and max_steps <= 0):
        raise ValueError("repeat and max_steps must be positive")
    action = np.clip(np.asarray(action), -1.0, 1.0)
    limit = min(repeat, max_steps) if max_steps is not None else repeat
    discounted_reward = raw_reward = energy = 0.0
    max_position = -float("inf")
    max_speed = 0.0
    for i in range(limit):
        obs, reward, terminated, truncated, _ = env.step(action)
        discounted_reward += gamma ** i * reward
        raw_reward += reward
        energy += float(np.square(action).sum())
        max_position = max(max_position, float(obs[0]))
        max_speed = max(max_speed, abs(float(obs[1])))
        if terminated or truncated:
            break
    return obs, discounted_reward, terminated, truncated, {"duration": i + 1, "raw_reward": raw_reward, "action_energy": energy, "max_position": max_position, "max_speed": max_speed}
