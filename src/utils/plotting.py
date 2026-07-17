"""
Styling and plotting utilities for generating publication-ready scientific figures.
"""

from typing import List, Dict, Any, Optional
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from src.configs.config import FIGURES_DIR


def set_scientific_style() -> None:
    """Sets matplotlib parameters to standard publication-ready styles."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'legend.fontsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'axes.grid': True,
        'grid.alpha': 0.25,
        'grid.linestyle': '--',
        'axes.edgecolor': '#333333',
        'axes.linewidth': 0.8,
    })


def set_alternative_style() -> None:
    """Alternative style setting (with serif fonts) as in some notebook cells."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 12,
        "axes.labelsize": 14,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "figure.figsize": (8, 6),
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.4,
        "grid.linestyle": "--"
    })


def plot_combined_learning_curves(
    all_rewards: Dict[int, List[float]],
    output_path: Optional[str] = None,
    prefix: str = "mappo"
) -> None:
    """Plots and saves the learning curve across multiple training seeds."""
    set_scientific_style()
    plt.figure(figsize=(9, 5))

    # Convert rewards dict to 2D numpy array
    seeds = sorted(list(all_rewards.keys()))
    runs = np.array([all_rewards[s] for s in seeds])  # Shape: (num_seeds, num_episodes)
    mean_curve = np.mean(runs, axis=0)
    std_curve = np.std(runs, axis=0)

    episodes = np.arange(1, len(mean_curve) + 1)
    plt.plot(episodes, mean_curve, color='#1f77b4', linewidth=1.8, label=f'{prefix.upper()} (Mean)')
    plt.fill_between(episodes, mean_curve - std_curve, mean_curve + std_curve, color='#1f77b4', alpha=0.15, label='±1 Std Dev')

    plt.xlabel('Training Episode', fontweight='bold')
    plt.ylabel('Episode Reward', fontweight='bold')
    plt.title(f'Multi-Seed Learning Curve ({len(seeds)} Seeds)', fontweight='bold', pad=10)
    plt.legend(loc='lower right', frameon=True)
    plt.grid(True, alpha=0.3)

    if output_path is None:
        os.makedirs(FIGURES_DIR, exist_ok=True)
        output_path = os.path.join(FIGURES_DIR, f"{prefix}_combined_learning_curve.png")

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 Learning curves plot saved: {output_path}")
