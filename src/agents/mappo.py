"""
Multi-Agent PPO (MAPPO) Policy Implementation.
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
    PROCUREMENT_ACTION_DIM,
    LOGISTICS_ACTION_DIM,
    STATE_SCALE
)
from src.agents.models import Actor, RunningMeanStd


class MultiAgentMAPPO:
    """Multi-Agent MAPPO policy with procurement and logistics actor heads."""

    def __init__(self, device: torch.device = DEVICE):
        self.device = device
        self.proc_actor = Actor(PROCUREMENT_ACTION_DIM).to(self.device)
        self.log_actor = Actor(LOGISTICS_ACTION_DIM).to(self.device)
        self.critic = nn.Sequential(
            nn.Linear(STATE_DIM, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        ).to(self.device)

        self.opt_p = Adam(self.proc_actor.parameters(), lr=LR)
        self.opt_l = Adam(self.log_actor.parameters(), lr=LR)
        self.opt_c = Adam(self.critic.parameters(), lr=LR)
        self.entropy_coef = ENTROPY_COEF
        self.rms = RunningMeanStd()

    def _norm(self, st: torch.Tensor) -> torch.Tensor:
        st = st.clone()
        st[:, :10] /= STATE_SCALE
        return st

    def get_action(self, state: np.ndarray) -> tuple:
        st = self._norm(torch.FloatTensor(state).unsqueeze(0).to(self.device))
        ap, lp = self.proc_actor.get_action(st)
        al, ll = self.log_actor.get_action(st)
        return np.concatenate([ap, al]), ap, al, lp, ll

    def get_deterministic_joint_action(self, state: np.ndarray) -> np.ndarray:
        st = self._norm(torch.FloatTensor(state).unsqueeze(0).to(self.device))
        with torch.no_grad():
            act_proc = self.proc_actor.forward(st).mean.squeeze(0).cpu().numpy()
            act_log = self.log_actor.forward(st).mean.squeeze(0).cpu().numpy()
        return np.concatenate([act_proc, act_log], axis=-1)

    def save(self, path: str) -> None:
        torch.save({
            'proc_actor': self.proc_actor.state_dict(),
            'log_actor': self.log_actor.state_dict(),
            'critic': self.critic.state_dict()
        }, path)

    def load(self, path: str) -> None:
        ck = torch.load(path, map_location=self.device, weights_only=True)
        self.proc_actor.load_state_dict(ck['proc_actor'])
        self.log_actor.load_state_dict(ck['log_actor'])
        self.critic.load_state_dict(ck['critic'])

    def update(self, s, ap, al, lp, ll, r, ns, d, masks) -> None:
        s = self._norm(torch.FloatTensor(np.array(s)).to(self.device))
        ns = self._norm(torch.FloatTensor(np.array(ns)).to(self.device))
        ap = torch.FloatTensor(np.array(ap)).to(self.device)
        al = torch.FloatTensor(np.array(al)).to(self.device)
        lp = torch.FloatTensor(np.array(lp)).to(self.device)
        ll = torch.FloatTensor(np.array(ll)).to(self.device)

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
            new_logp_p, ent_p = self.proc_actor.evaluate(s, ap)
            ratio_p = torch.exp(new_logp_p - lp.detach())
            surr1_p = ratio_p * adv.unsqueeze(1)
            surr2_p = torch.clamp(ratio_p, 1 - EPSILON, 1 + EPSILON) * adv.unsqueeze(1)
            loss_p = -torch.min(surr1_p, surr2_p) * masks
            loss_p = loss_p.sum() / (masks.sum() + 1e-8)
            loss_p -= self.entropy_coef * (ent_p * masks).sum() / (masks.sum() + 1e-8)

            self.opt_p.zero_grad()
            loss_p.backward()
            nn.utils.clip_grad_norm_(self.proc_actor.parameters(), 0.5)
            self.opt_p.step()

            new_logp_l, ent_l = self.log_actor.evaluate(s, al)
            ratio_l = torch.exp(new_logp_l - ll.detach())
            surr1_l = ratio_l * adv.unsqueeze(1)
            surr2_l = torch.clamp(ratio_l, 1 - EPSILON, 1 + EPSILON) * adv.unsqueeze(1)
            loss_l = -torch.min(surr1_l, surr2_l).mean() - self.entropy_coef * ent_l.mean()

            self.opt_l.zero_grad()
            loss_l.backward()
            nn.utils.clip_grad_norm_(self.log_actor.parameters(), 0.5)
            self.opt_l.step()

            v_pred = self.critic(s).squeeze()
            loss_c = F.mse_loss(v_pred, returns)
            self.opt_c.zero_grad()
            loss_c.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.opt_c.step()

        self.entropy_coef = max(MIN_ENTROPY, self.entropy_coef * 0.999)
