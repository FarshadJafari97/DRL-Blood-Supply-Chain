"""
Unit tests for the platelet inventory reinforcement learning codebase.
Tests the environment transitions, demand sampling, and baselines.
"""

import sys
import os
import numpy as np
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.env.platelet_env import PlateletSupplyChainEnv
from src.demand.demand_model import sample_hospital_demand, compute_zinb_mean
from src.baselines.order_up_to import OrderUpToBaseline


def test_demand_mean_calculation():
    """Verifies ZINB analytical mean calculator."""
    params = {"phi": 0.2, "p": 0.5, "n": 10}
    # mean = (1 - 0.2) * 10 * (1 - 0.5) / 0.5 = 0.8 * 10 * 1.0 = 8.0
    expected = 8.0
    assert abs(compute_zinb_mean(params) - expected) < 1e-8


def test_sample_demand():
    """Verifies hospital demand sampler works under base conditions."""
    params = [{"phi": 0.25, "p": 0.48, "n": 15}]
    demands = [sample_hospital_demand(params) for _ in range(50)]
    assert all(isinstance(d, int) for d in demands)
    assert all(d >= 0 for d in demands)


def test_env_initialization():
    """Ensures env initializes with correct capacities, shelf life, and starting arrays."""
    env = PlateletSupplyChainEnv()
    assert env.num_hospitals == 2
    assert env.state.shape == (10,)
    assert np.sum(env.state) == 0.0
    assert env.day == 0


def test_env_step():
    """Tests environment state, step transition, reward, done, and info dictionary layout."""
    env = PlateletSupplyChainEnv()
    obs = env.reset()
    assert obs.shape == (26,)

    # Test order day actions
    action = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    next_obs, reward, done, info = env.step(action)

    assert next_obs.shape == (26,)
    assert isinstance(reward, (float, np.floating))
    assert isinstance(done, bool)
    assert "demand_h1" in info
    assert "demand_h2" in info
    assert "total_cost" in info


def test_order_up_to_baseline():
    """Tests baseline action generation."""
    env = PlateletSupplyChainEnv()
    obs = env.reset()
    policy = OrderUpToBaseline(S1=60, S2=60)
    action = policy.get_action(env, obs)
    assert len(action) == 3
    assert action[0] >= 0.0
    assert action[1] >= 0.0
    assert action[2] == 0.5
