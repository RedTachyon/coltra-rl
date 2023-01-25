from __future__ import annotations

from typing import Callable

import numpy as np

from coltra.data_utils import Trajectory


# def evaluate_trajectory(trajectory: Trajectory, reward_function: Callable[[np.ndarray, np.ndarray], float], gamma: float = 0.99, eta: float = 0.0) -> float:
#     """Evaluate a trajectory according to the reward function.
#
#     Args:
#         trajectory (Trajectory): Trajectory to evaluate.
#         reward_function:
#         discount (float): Discount factor.
#
#     Returns:
#         float: Trajectory evaluation.
#     """
#     pass
#
#
# def annotate_actions(trajectory: Trajectory, dynamics: str = "CarVel") -> np.ndarray:
#     """Annotate the actions of a trajectory with the next state, next action, and reward.
#
#     Args:
#         trajectory (Trajectory): Trajectory to annotate.
#         dynamics (str, optional): Dynamics model to use. Defaults to "CarVel".
#
#     Returns:
#         Trajectory: Annotated trajectory.
#     """
#     pass
