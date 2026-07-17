"""
Random seed handling helpers for reproducibility.
"""

import random
import numpy as np
import torch


def set_seed(seed: int) -> None:
    """
    Sets the random seed for Python, Numpy, and PyTorch (including CUDA if available).

    Args:
        seed: The seed integer.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
