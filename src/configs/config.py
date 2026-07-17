"""
Configuration and hyperparameters for platelet supply chain reinforcement learning.
Includes environment parameters, demand specs, policy definitions, and training configs.
"""

import os
import torch

# General Device Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Output Configuration
RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../results"))
MODELS_DIR = os.path.join(RESULTS_DIR, "models")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")
LOGS_DIR = os.path.join(RESULTS_DIR, "logs")

# Demand Specifications
ZINB_H1_PARAMS = [{"phi": 0.25, "p": 0.48, "n": 15}]
ZINB_H2_PARAMS = [{"phi": 0.25, "p": 0.48, "n": 15}]

# Base Environment Configurations
ENV_CONFIG = {
    "max_capacity": 100,
    "shelf_life": 5,
    "order_cost": 1.0,
    "transshipment_cost": 1.0,
    "transshipment_fixed_cost": 5.0,
    "shortage_cost": 25.0,
    "wastage_cost": 15.0,
    "inventory_cost": 0.5,
    "episode_length": 60,
    "order_cap": 70,
    "trans_cap": 50,
    "transshipment_deadband": 0.1,
    "disruption_threshold": 1.2,
    "disruption_prob": 0.02,
    "disruption_min_len": 3,
    "disruption_max_len": 8,
    "disruption_mult": 1.4,
}

# Order Day Schedules for H1 and H2
ORDER_SCHEDULE_H1 = {1, 4, 6}
ORDER_SCHEDULE_H2 = {0, 3, 5}

# State and Action Dimensions
STATE_DIM = 26
PROCUREMENT_ACTION_DIM = 2
LOGISTICS_ACTION_DIM = 1
STATE_SCALE = 70.0
REWARD_SCALE = 100.0

# Base RL Algorithm Hyperparameters
LR = 3e-4
GAMMA = 0.99
EPSILON = 0.2
EPOCHS = 4
GAE_LAMBDA = 0.95
UPDATE_EPISODES = 4
ENTROPY_COEF = 0.05
ENTROPY_DECAY = 0.999
MIN_ENTROPY = 0.005
TOTAL_EPISODES = 2000

# Training seed definitions
TRAIN_SEEDS = [42, 123, 456, 789, 2025]
EVAL_SEEDS = range(1, 51)
