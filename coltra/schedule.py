from __future__ import annotations

from typing import Dict, Union, Any, Callable, Optional

import numpy as np

INT_MAX = 2**31 - 1

BASE_CONFIG: dict[str, int | str | float | bool | None] = {
    "num_agents": 20,
    "initializer": "Circle",
    "dynamics": "PolarAcceleration",
    "observer": "Egocentric",
    "rewarder": "BaseRewarder",
    "potential": 1,
    "goal": 10,
    "collision": -0.05,
    "step_reward": -0.005,
    "standstill_weight": 0,
    "standstill_exponent": 0,
    "goal_speed_threshold": 0,
    "comfort_speed": 1.33,
    "comfort_speed_weight": -0.75,
    "comfort_speed_exponent": 1,
    "comfort_distance": 0,
    "comfort_distance_weight": 0,
    "energy_weight": 1,
    "final_energy_weight": 1,
    "sight_radius": 10,
    "sight_agents": 50,
    "sight_angle": 135,
    "sight_acceleration": False,
    "rays_per_direction": 10,
    "ray_length": 10,
    "ray_degrees": 90,
    "ray_agent_vision": False,
    "destroy_raycasts": False,
    "spawn_noise_scale": 0.3,
    "spawn_scale": 6,
    "grid_spawn": True,
    "group_spawn_scale": 1.5,
    "enable_obstacles": False,
    "block_scale": 3,
    "random_mass": True,
    "random_energy": True,
    "shared_goal": False,
    "evaluation_mode": False,
    "early_finish": False,
    "nice_colors": True,
    "show_attention": False,
    "backwards_allowed": True,
}

SCENARIOS = ["Circle", "CircleBlock" "Corridor", "Crossway", "Random"]

Curriculum = Callable[[Optional[dict], int, int], dict]

SimpleCurriculum = dict[int, dict[str, Any]]


def fixed_curriculum(config: dict | None = None, seed: int = 0, step: int = 0) -> dict:
    """
    Uses a fixed environment configuration. Effectively a no-op to fit the template.
    """
    if config is None:
        config = BASE_CONFIG
    config = config.copy()
    return config


def random_curriculum(config: dict | None = None, seed: int = 0, step: int = 0) -> dict:
    """
    Samples an environment configuration.
    Based on the default config, but randomly samples the configuration and number of agents

    To randomize:
    - Scenario (`initializer`): categorical
    - Number of agents (`num_agents`): 5 - 50
    - Obstacles? (`enable_obstacles`): binary
    """
    if config is None:
        config = BASE_CONFIG

    config = config.copy()
    _rng = np.random.default_rng(step)
    seed = seed + _rng.integers(0, INT_MAX)

    rng = np.random.default_rng(seed)

    scenario = rng.choice(SCENARIOS)
    num_agents = rng.integers(5, 50)
    obstacles = rng.choice([True, False])

    config["initializer"] = scenario
    config["num_agents"] = num_agents
    config["enable_obstacles"] = obstacles

    return config
