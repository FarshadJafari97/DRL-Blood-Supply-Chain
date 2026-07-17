"""
Demand generation functions for hospital platelet demand simulation.
Includes Zero-Inflated Negative Binomial (ZINB) sampling and mean calculations.
"""

from typing import List, Dict, Any
import numpy as np


def compute_zinb_mean(params: Dict[str, Any]) -> float:
    """
    Computes the analytical mean of the Zero-Inflated Negative Binomial (ZINB) distribution.

    Args:
        params: Dictionary containing 'phi' (zero inflation probability),
                'p' (probability of success), and 'n' (number of successes).

    Returns:
        The expected value of the ZINB distribution.
    """
    phi, p, n = params["phi"], params["p"], params["n"]
    return float((1.0 - phi) * n * (1.0 - p) / p)


def sample_zinb(phi: float, p: float, n: int) -> int:
    """
    Draws a single sample from a Zero-Inflated Negative Binomial (ZINB) distribution.

    Args:
        phi: Probability of getting a zero (inflation factor).
        p: Negative binomial success probability.
        n: Number of successes parameter.

    Returns:
        A discrete sample value.
    """
    if np.random.random() < phi:
        return 0
    return int(np.random.negative_binomial(n, p))


def sample_hospital_demand(zinb_params: List[Dict[str, Any]], mult: float = 1.0) -> int:
    """
    Samples total demand for a hospital by summing ZINB component samples and applying a multiplier.

    Args:
        zinb_params: List of dictionaries representing different ZINB demand components.
        mult: A multiplier factor (e.g. to simulate demand spikes during disruption periods).

    Returns:
        The total simulated hospital demand (clamped at a minimum of 0).
    """
    total = sum(sample_zinb(p["phi"], p["p"], p["n"]) for p in zinb_params)
    return max(0, int(total * mult))
