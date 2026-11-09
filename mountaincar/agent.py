"""Gaussian policy and value function for continuous Mountain Car."""

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal


def clip_action(action):
    return np.clip(action, -1.0, 1.0)


def _network(output_size):
    return nn.Sequential(nn.Linear(2, 64), nn.Tanh(), nn.Linear(64, 64),
                         nn.Tanh(), nn.Linear(64, output_size))


class Agent(nn.Module):
    def __init__(self):
        super().__init__()
        self.action_repeat = 1
        self.actor = _network(1)
        self.critic = _network(1)
        self.log_std = nn.Parameter(torch.zeros(1))
        self.register_buffer("obs_center", torch.tensor([-0.3, 0.0]))
        self.register_buffer("obs_scale", torch.tensor([0.9, 0.07]))

    def _scale(self, obs):
        return (obs - self.obs_center) / self.obs_scale

    def distribution(self, obs):
        mean = self.actor(self._scale(obs))
        return Normal(mean, self.log_std.exp().expand_as(mean))

    def value(self, obs):
        return self.critic(self._scale(obs)).squeeze(-1)

    def act(self, obs, deterministic=False):
        distribution = self.distribution(obs)
        action = distribution.mean if deterministic else distribution.sample()
        return action, distribution.log_prob(action).sum(-1), self.value(obs)

    def save(self, path, metadata=None):
        torch.save({"state_dict": self.state_dict(), "metadata": {**(metadata or {}), "action_repeat": self.action_repeat}}, Path(path))

    @classmethod
    def load(cls, path):
        checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
        agent = cls()
        agent.load_state_dict(checkpoint["state_dict"])
        agent.action_repeat = checkpoint.get("metadata", {}).get("action_repeat", 1)
        agent.eval()
        return agent
