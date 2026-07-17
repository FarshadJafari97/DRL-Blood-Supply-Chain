"""
Baseline inventory control policies.
Includes Order-Up-To policy (S1, S2).
"""

import numpy as np
from src.env.platelet_env import PlateletSupplyChainEnv


class OrderUpToBaseline:
    """
    Order-Up-To (S1, S2) inventory baseline policy.
    Maintains target inventory levels S1 and S2 at the hospitals.
    """

    def __init__(self, S1: float = 120.0, S2: float = 120.0):
        self.S1 = S1
        self.S2 = S2

    def get_action(self, env: PlateletSupplyChainEnv, obs: np.ndarray) -> np.ndarray:
        """
        Determines procurement actions based on inventory shortfall from target level.

        Args:
            env: The PlateletSupplyChainEnv instance.
            obs: The current observation array.

        Returns:
            An action array containing procurement proportions for H1 and H2.
            The third dimension is a neutral action for compatibility when needed.
        """
        # Determine total on-hand inventory for each hospital
        # Note: on-hand inventory includes items at ages 0 to shelf_life-1
        inv_h1 = np.sum(env.state[0 : env.shelf_life])
        inv_h2 = np.sum(env.state[env.shelf_life : env.shelf_life * 2])

        # Shortfall from the target Order-Up-To levels
        shortfall_1 = max(0.0, self.S1 - inv_h1)
        shortfall_2 = max(0.0, self.S2 - inv_h2)

        # Convert shortfalls to ratios of order cap
        # (Clamped to 0.0-1.0 as the environment will clip actions inside step)
        act_1 = shortfall_1 / env.order_cap
        act_2 = shortfall_2 / env.order_cap

        # No lateral transshipment is made in this baseline;
        # Return a neutral transshipment action (0.5 results in 0.0 transshipment)
        return np.array([act_1, act_2, 0.5], dtype=np.float32)
