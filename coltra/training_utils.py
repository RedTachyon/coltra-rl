from typing import Optional

import numpy as np
from tqdm import trange

from coltra import MultiAgentEnv
from coltra.groups import MacroAgent


def evaluate(
    env: MultiAgentEnv,
    group: MacroAgent,
    n_episodes: int = 5,
    n_steps: int = 200,
    disable_tqdm: bool = False,
    reset_kwargs: Optional[dict] = None,
) -> tuple[np.ndarray, np.ndarray]:
    if reset_kwargs is None:
        reset_kwargs = {}

    returns = []
    energies = []
    current_return = 0.0
    for ep in trange(n_episodes, disable=disable_tqdm):
        obs = env.reset(**reset_kwargs)
        for t in range(n_steps):
            action, _, _ = group.act(obs)
            obs, reward, done, info = env.step(action)

            mean_reward = np.mean(list(reward.values()))
            current_return += mean_reward
            if all(done.values()):
                returns.append(current_return)
                energies.append(info["e_energy"])
                current_return = 0.0
                break
    return np.array(returns), np.array(energies)
