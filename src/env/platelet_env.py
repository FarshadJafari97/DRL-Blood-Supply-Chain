"""
Platelet Supply Chain Environment.
A simulation environment for managing platelet inventory across hospital networks.
Supports both procurement orders and optional lateral transshipment actions.
"""

import random
from collections import deque
from typing import Dict, Any, Tuple, List, Optional
import numpy as np

from src.configs.config import (
    ENV_CONFIG,
    ZINB_H1_PARAMS,
    ZINB_H2_PARAMS,
    ORDER_SCHEDULE_H1,
    ORDER_SCHEDULE_H2,
    STATE_DIM,
    REWARD_SCALE
)
from src.demand.demand_model import compute_zinb_mean, sample_hospital_demand


class PlateletSupplyChainEnv:
    """
    Simulation Environment for Platelet Supply Chain Inventory Management.
    Supports 2 hospitals with custom age-based platelet inventory states.
    """

    def __init__(self, env_config: Optional[Dict[str, Any]] = None, enable_transshipment: bool = True):
        """
        Initializes the Platelet Supply Chain Environment.

        Args:
            env_config: Custom configuration dictionary. If None, uses default ENV_CONFIG.
            enable_transshipment: Whether to process and allow lateral transshipments between hospitals.
        """
        config = ENV_CONFIG.copy()
        if env_config is not None:
            config.update(env_config)

        for k, v in config.items():
            setattr(self, k, v)

        self.num_hospitals = 2
        self.enable_transshipment = enable_transshipment
        self.demand_mean_h1 = sum(compute_zinb_mean(p) for p in ZINB_H1_PARAMS)
        self.demand_mean_h2 = sum(compute_zinb_mean(p) for p in ZINB_H2_PARAMS)
        self.reset()

    def _is_order_day_h1(self) -> bool:
        """Checks if today is an order day for Hospital 1."""
        return (self.day % 7) in ORDER_SCHEDULE_H1

    def _is_order_day_h2(self) -> bool:
        """Checks if today is an order day for Hospital 2."""
        return (self.day % 7) in ORDER_SCHEDULE_H2

    def _compute_disruption_flag(self, history: List[int]) -> float:
        """
        Computes the disruption flag based on moving averages of demand history.

        Args:
            history: Demand history list.

        Returns:
            1.0 if a disruption is detected, 0.0 otherwise.
        """
        if len(history) < 5:
            return 0.0
        ma2 = np.mean(list(history)[-2:])
        ma30 = np.mean(list(history)[-30:])
        return 1.0 if (ma30 > 0 and ma2 > self.disruption_threshold * ma30) else 0.0

    def _get_observation(self) -> np.ndarray:
        """
        Constructs the current observation state vector of dimension STATE_DIM.

        Returns:
            A numpy array representing the normalized state observation.
        """
        obs = np.zeros((STATE_DIM,), dtype=np.float32)
        idx = 0
        for h in range(self.num_hospitals):
            start = h * self.shelf_life
            for age in range(self.shelf_life):
                obs[idx] = self.state[start + age]
                idx += 1

        inv_h1 = np.sum(self.state[0:self.shelf_life])
        inv_h2 = np.sum(self.state[self.shelf_life : self.shelf_life * 2])

        obs[10] = self.day / self.episode_length
        obs[11] = self._compute_disruption_flag(self.demand_history_h1)
        obs[12] = self._compute_disruption_flag(self.demand_history_h2)
        obs[13] = inv_h1 / self.max_capacity
        obs[14] = inv_h2 / self.max_capacity
        obs[15] = 1.0 if self._is_order_day_h1() else 0.0
        obs[16] = 1.0 if self._is_order_day_h2() else 0.0
        obs[17] = (inv_h1 - inv_h2) / self.max_capacity
        obs[18] = self.crisis_counter_h1 / self.disruption_max_len
        obs[19] = self.crisis_counter_h2 / self.disruption_max_len

        d1 = np.mean(self.demand_history_h1[-7:]) if self.demand_history_h1 else self.demand_mean_h1
        d2 = np.mean(self.demand_history_h2[-7:]) if self.demand_history_h2 else self.demand_mean_h2
        obs[20] = min(inv_h1 / (d1 * 7 + 1e-8), 3.0)
        obs[21] = min(inv_h2 / (d2 * 7 + 1e-8), 3.0)

        if self.demand_history_h1:
            td1 = sum(self.demand_history_h1[-30:])
            obs[22] = sum(self.shortage_history_h1) / (td1 + 1e-8)
            obs[24] = sum(self.wastage_history_h1) / (td1 + 1e-8)
        if self.demand_history_h2:
            td2 = sum(self.demand_history_h2[-30:])
            obs[23] = sum(self.shortage_history_h2) / (td2 + 1e-8)
            obs[25] = sum(self.wastage_history_h2) / (td2 + 1e-8)
        return obs.copy()

    def reset(self) -> np.ndarray:
        """
        Resets the environment state and history buffers.

        Returns:
            The initial state observation vector.
        """
        self.state = np.zeros((self.num_hospitals * self.shelf_life,), dtype=np.float32)
        self.day = 0
        self.demand_history_h1, self.demand_history_h2 = [], []
        self.shortage_history_h1 = deque(maxlen=30)
        self.shortage_history_h2 = deque(maxlen=30)
        self.wastage_history_h1 = deque(maxlen=30)
        self.wastage_history_h2 = deque(maxlen=30)
        self.disruption_counter_h1 = self.disruption_counter_h2 = 0
        self.crisis_counter_h1 = self.crisis_counter_h2 = 0
        return self._get_observation()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Steps the simulation forward by one day.

        Args:
            action: Action vector containing procurement (and logistics) ratios.

        Returns:
            A tuple of (next_observation, joint_reward, done, info_dict)
        """
        action = np.clip(action, 0.0, 1.0)

        is_ord_h1 = self._is_order_day_h1()
        is_ord_h2 = self._is_order_day_h2()

        order_h1 = action[0] * self.order_cap if is_ord_h1 else 0.0
        order_h2 = action[1] * self.order_cap if is_ord_h2 else 0.0

        # Optional Lateral Transshipment Handling
        actual_trans_h1_to_h2 = 0.0
        actual_trans_h2_to_h1 = 0.0
        if self.enable_transshipment and len(action) > 2:
            a_t = action[2] * 2.0 - 1.0
            tau = self.transshipment_deadband
            trans_h1_to_h2 = 0.0
            trans_h2_to_h1 = 0.0
            if a_t > tau:
                trans_h1_to_h2 = ((a_t - tau) / (1.0 - tau)) * self.trans_cap
            elif a_t < -tau:
                trans_h2_to_h1 = ((abs(a_t) - tau) / (1.0 - tau)) * self.trans_cap

        inv_h1_before = np.sum(self.state[0 : self.shelf_life])
        inv_h2_before = np.sum(self.state[self.shelf_life : self.shelf_life * 2])

        actual_order_h1 = self._process_order(
            0, min(order_h1, max(0.0, self.max_capacity - inv_h1_before))
        )
        actual_order_h2 = self._process_order(
            1, min(order_h2, max(0.0, self.max_capacity - inv_h2_before))
        )

        if self.enable_transshipment and len(action) > 2:
            actual_trans_h1_to_h2 = self._transfer_blood(0, 1, trans_h1_to_h2)
            actual_trans_h2_to_h1 = self._transfer_blood(1, 0, trans_h2_to_h1)

        # Disruption updates
        if self.day >= 5:
            if self.disruption_counter_h1 == 0 and random.random() < self.disruption_prob:
                self.disruption_counter_h1 = random.randint(self.disruption_min_len, self.disruption_max_len)
            if self.disruption_counter_h2 == 0 and random.random() < self.disruption_prob:
                self.disruption_counter_h2 = random.randint(self.disruption_min_len, self.disruption_max_len)

        mult_h1 = self.disruption_mult if self.disruption_counter_h1 > 0 else 1.0
        mult_h2 = self.disruption_mult if self.disruption_counter_h2 > 0 else 1.0

        if self.disruption_counter_h1 > 0:
            self.disruption_counter_h1 -= 1
        if self.disruption_counter_h2 > 0:
            self.disruption_counter_h2 -= 1

        demand_h1 = sample_hospital_demand(ZINB_H1_PARAMS, mult=mult_h1)
        demand_h2 = sample_hospital_demand(ZINB_H2_PARAMS, mult=mult_h2)

        self.demand_history_h1.append(demand_h1)
        self.demand_history_h2.append(demand_h2)

        self.crisis_counter_h1 = (
            min(self.crisis_counter_h1 + 1, self.disruption_max_len)
            if self._compute_disruption_flag(self.demand_history_h1) > 0.5
            else 0
        )
        self.crisis_counter_h2 = (
            min(self.crisis_counter_h2 + 1, self.disruption_max_len)
            if self._compute_disruption_flag(self.demand_history_h2) > 0.5
            else 0
        )

        wastage_h1, shortage_h1, _ = self._fulfill_demand(0, demand_h1)
        wastage_h2, shortage_h2, _ = self._fulfill_demand(1, demand_h2)

        self.shortage_history_h1.append(shortage_h1)
        self.shortage_history_h2.append(shortage_h2)
        self.wastage_history_h1.append(wastage_h1)
        self.wastage_history_h2.append(wastage_h2)

        self._age_blood()

        inv_h1_after = np.sum(self.state[0 : self.shelf_life])
        inv_h2_after = np.sum(self.state[self.shelf_life : self.shelf_life * 2])

        inventory_holding_cost = (inv_h1_before + inv_h2_before) * self.inventory_cost
        self.day += 1

        total_transferred = actual_trans_h1_to_h2 + actual_trans_h2_to_h1
        if self.enable_transshipment:
            transshipment_cost = (
                total_transferred * self.transshipment_cost
                + (self.transshipment_fixed_cost if total_transferred >= 1.0 else 0.0)
            )
        else:
            transshipment_cost = 0.0

        wastage_cost = (wastage_h1 + wastage_h2) * self.wastage_cost
        shortage_cost = (shortage_h1 + shortage_h2) * self.shortage_cost
        order_cost = (actual_order_h1 + actual_order_h2) * self.order_cost

        total_cost = wastage_cost + shortage_cost + transshipment_cost + order_cost + inventory_holding_cost
        shared_reward = -total_cost / REWARD_SCALE

        info = {
            "order_mask": [1.0 if is_ord_h1 else 0.0, 1.0 if is_ord_h2 else 0.0],
            "demand_h1": demand_h1,
            "demand_h2": demand_h2,
            "inventory_h1": inv_h1_after,
            "inventory_h2": inv_h2_after,
            "order_h1": order_h1,
            "order_h2": order_h2,
            "actual_order_h1": actual_order_h1,
            "actual_order_h2": actual_order_h2,
            "transferred_h1_to_h2": actual_trans_h1_to_h2,
            "transferred_h2_to_h1": actual_trans_h2_to_h1,
            "shortage_h1": shortage_h1,
            "shortage_h2": shortage_h2,
            "wastage_h1": wastage_h1,
            "wastage_h2": wastage_h2,
            "total_cost": total_cost,
            "wastage_cost": wastage_cost,
            "shortage_cost": shortage_cost,
            "transshipment_cost": transshipment_cost,
            "order_cost": order_cost,
            "inventory_holding_cost": inventory_holding_cost,
            "disruption_active_h1": mult_h1 > 1.0,
            "disruption_active_h2": mult_h2 > 1.0,
            "crisis_counter_h1": self.crisis_counter_h1,
            "crisis_counter_h2": self.crisis_counter_h2,
            "inv_h1_before": inv_h1_before,
            "inv_h2_before": inv_h2_before,
        }

        done = self.day >= self.episode_length
        return self._get_observation(), shared_reward, done, info

    def _process_order(self, hospital: int, amount: float) -> float:
        """Processes platelet replenishment orders for a hospital."""
        if amount <= 0:
            return 0.0
        start = hospital * self.shelf_life
        accepted = max(
            0.0,
            min(
                amount,
                self.max_capacity - np.sum(self.state[start : start + self.shelf_life]),
            ),
        )
        self.state[start] += accepted
        return accepted

    def _transfer_blood(self, from_h: int, to_h: int, amount: float) -> float:
        """Transfers blood units laterally from one hospital to another."""
        if amount <= 0:
            return 0.0
        f_s, f_e = from_h * self.shelf_life, from_h * self.shelf_life + self.shelf_life
        t_s, t_e = to_h * self.shelf_life, to_h * self.shelf_life + self.shelf_life

        transferrable = max(
            0.0,
            min(
                amount,
                np.sum(self.state[f_s:f_e]),
                self.max_capacity - np.sum(self.state[t_s:t_e]),
            ),
        )
        transferred = 0.0
        for i in range(f_e - 1, f_s - 1, -1):
            if self.state[i] > 0:
                to_transfer = min(self.state[i], transferrable - transferred)
                self.state[i] -= to_transfer
                self.state[t_s + (i - f_s)] += to_transfer
                transferred += to_transfer
                if transferred >= transferrable:
                    break
        return transferred

    def _fulfill_demand(self, hospital: int, demand: int) -> Tuple[float, float, float]:
        """Fulfills demand using oldest platelets first and gets wastage."""
        s_idx, e_idx = hospital * self.shelf_life, hospital * self.shelf_life + self.shelf_life
        unfulfilled = demand
        for i in range(e_idx - 1, s_idx - 1, -1):
            if self.state[i] > 0 and unfulfilled > 0:
                used = min(self.state[i], unfulfilled)
                self.state[i] -= used
                unfulfilled -= used
            if unfulfilled <= 1e-5:
                break
        wastage = self.state[e_idx - 1]
        self.state[e_idx - 1] = 0
        return wastage, unfulfilled, demand - unfulfilled

    def _age_blood(self) -> None:
        """Ages all blood inside the environment by moving ages forward."""
        for h in range(self.num_hospitals):
            s_idx, e_idx = h * self.shelf_life, h * self.shelf_life + self.shelf_life
            self.state[s_idx + 1 : e_idx] = self.state[s_idx : e_idx - 1]
            self.state[s_idx] = 0
