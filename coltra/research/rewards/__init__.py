from coltra.data_utils import Trajectory


def evaluate_trajectory(trajectory: Trajectory, reward_function: callable, gamma: float = 0.99, eta: float = 0.0) -> float:
    """Evaluate a trajectory according to the reward function.

    Args:
        trajectory (Trajectory): Trajectory to evaluate.
        reward_function:
        discount (float): Discount factor.

    Returns:
        float: Trajectory evaluation.
    """
