"""Record one deterministic episode."""
from pathlib import Path
import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
from .evaluation import action_repeat, choose_action


def record(policy, output, seed=20000):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    env = gym.make("MountainCarContinuous-v0", render_mode="rgb_array")
    rng = np.random.default_rng(seed)
    try:
        observation, _ = env.reset(seed=seed)
        with imageio.get_writer(str(output), fps=env.metadata.get("render_fps", 30)) as writer:
            writer.append_data(env.render())
            done = False
            length = 0
            while not done and length < 999:
                action = choose_action(policy, observation, rng)
                for _ in range(min(action_repeat(policy), 999 - length)):
                    observation, _, terminated, truncated, _ = env.step(action)
                    writer.append_data(env.render())
                    length += 1
                    done = terminated or truncated
                    if done:
                        break
    finally:
        env.close()
    return output
