"""
Single-Agent PPO Policy Implementation.
Supports configurations with or without transshipment actions.
"""

from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from src.configs.config import (
    DEVICE,
    LR,
    GAMMA,
    EPSILON,
    EPOCHS,
    GAE_LAMBDA,
    ENTROPY_COEF,
    MIN_ENTROPY,
    STATE_DIM,
    STATE_SCALE
)
from src.agents.models import Actor, RunningMeanStd


class SingleAgentPPO:
    """Single Agent PPO Policy."""

    def __init__(self, action_dim: int, device: torch.device = DEVICE):
        """
        Initializes SingleAgentPPO.

        Args:
            action_dim: 3 if transshipment is enabled (SA-T), 2 if disabled (SA-NT).
            device: Training torch device.
        """
        self.device = device
        self.action_dim = action_dim
        self.actor = Actor(action_dim).to(self.device)
        self.critic = nn.Sequential(
            nn.Linear(STATE_DIM, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        ).to(self.device)

        self.opt_a = Adam(self.actor.parameters(), lr=LR)
        self.opt_c = Adam(self.critic.parameters(), lr=LR)
        self.entropy_coef = ENTROPY_COEF
        self.rms = RunningMeanStd()

    def _norm(self, st: torch.Tensor) -> torch.Tensor:
        st = st.clone()
        st[:, :10] /= STATE_SCALE
        return st

    def get_action(self, state: np.ndarray) -> tuple:
        st = self._norm(torch.FloatTensor(state).unsqueeze(0).to(self.device))
        act, logp = self.actor.get_action(st)
        return act, logp

    def get_deterministic_action(self, state: np.ndarray) -> np.ndarray:
        st = self._norm(torch.FloatTensor(state).unsqueeze(0).to(self.device))
        with torch.no_grad():
            act = self.actor.forward(st).mean.squeeze(0).cpu().numpy()
        return act

    def save(self, path: str) -> None:
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict()
        }, path)

    def load(self, path: str) -> None:
        ck = torch.load(path, map_location=self.device, weights_only=True)
        self.actor.load_state_dict(ck['actor'])
        self.critic.load_state_dict(ck['critic'])

    def update(self, s, a, lp, r, ns, d, masks) -> None:
        s = self._norm(torch.FloatTensor(np.array(s)).to(self.device))
        ns = self._norm(torch.FloatTensor(np.array(ns)).to(self.device))
        a = torch.FloatTensor(np.array(a)).to(self.device)
        lp = torch.FloatTensor(np.array(lp)).to(self.device)

        r_np = np.array(r)
        self.rms.update(r_np)
        r_norm = torch.FloatTensor(self.rms.normalize(r_np)).to(self.device)

        d = torch.FloatTensor(np.array(d)).to(self.device)
        masks = torch.FloatTensor(np.array(masks)).to(self.device)

        with torch.no_grad():
            vals = self.critic(s).squeeze()
            n_vals = self.critic(ns).squeeze()

        adv = torch.zeros_like(r_norm).to(self.device)
        gae = 0
        for t in reversed(range(len(r_norm))):
            delta = r_norm[t] + GAMMA * n_vals[t] * (1 - d[t]) - vals[t]
            gae = delta + GAMMA * GAE_LAMBDA * (1 - d[t]) * gae
            adv[t] = gae

        returns = adv + vals
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        for _ in range(EPOCHS):
            new_logp, ent = self.actor.evaluate(s, a)

            ratio = torch.exp(new_logp - lp.detach())

            # For SingleAgentPPO:
            # - If action_dim == 3 (SA-T), apply procurement mask on first 2 dimensions
            # - If action_dim == 2 (SA-NT), apply procurement mask on both/all dimensions
            full_mask = torch.ones_like(ratio)
            if self.action_dim == 3:
                full_mask[:, :2] = masks
            else:
                full_mask = masks

            surr1 = ratio * adv.unsqueeze(1)
            surr2 = torch.clamp(ratio, 1 - EPSILON, 1 + EPSILON) * adv.unsqueeze(1)

            loss_a = -torch.min(surr1, surr2) * full_mask
            loss_a = loss_a.sum() / (full_mask.sum() + 1e-8)
            loss_a -= self.entropy_coef * (ent * full_mask).sum() / (full_mask.sum() + 1e-8)

            self.opt_a.zero_grad()
            loss_a.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.opt_a.step()

            v_pred = self.critic(s).squeeze()
            loss_c = F.mse_loss(v_pred, returns)
            self.opt_c.zero_grad()
            loss_c.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.opt_c.step()

        self.entropy_coef = max(MIN_ENTROPY, self.entropy_coef * 0.999)
