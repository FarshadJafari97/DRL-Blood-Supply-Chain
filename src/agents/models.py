"""
Neural network components and normalization helpers for RL models.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta

from src.configs.config import STATE_DIM


class RunningMeanStd:
    """Tracks running mean and variance for reward normalization."""

    def __init__(self, epsilon: float = 1e-4, shape: tuple = ()):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x: np.ndarray) -> None:
        batch_mean = np.mean(x)
        batch_var = np.var(x)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean: float, batch_var: float, batch_count: int) -> None:
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)


class Actor(nn.Module):
    """Beta-distribution policy network for bounded continuous actions."""

    def __init__(self, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        self.alpha_head = nn.Linear(256, action_dim)
        self.beta_head = nn.Linear(256, action_dim)

    def forward(self, state: torch.Tensor) -> Beta:
        feat = self.net(state)
        alpha = torch.clamp(F.softplus(self.alpha_head(feat)) + 1.0, 1.0, 100.0)
        beta = torch.clamp(F.softplus(self.beta_head(feat)) + 1.0, 1.0, 100.0)
        return Beta(alpha, beta)

    def get_action(self, state: torch.Tensor) -> tuple:
        with torch.no_grad():
            dist = self.forward(state)
            action = dist.sample()
            safe_act = torch.clamp(action, 1e-5, 1.0 - 1e-5)
            return safe_act.squeeze(0).cpu().numpy(), dist.log_prob(safe_act).squeeze(0).cpu().numpy()

    def evaluate(self, state: torch.Tensor, action: torch.Tensor) -> tuple:
        dist = self.forward(state)
        safe_act = torch.clamp(action, 1e-5, 1.0 - 1e-5)
        return dist.log_prob(safe_act), dist.entropy()
