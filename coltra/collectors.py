from dataclasses import dataclass, field
from typing import Dict, Callable, List, Tuple, Optional, TypeVar, Union

import numpy as np
import torch
import torch.multiprocessing as mp
from tqdm import trange

from coltra.groups import MacroAgent
from coltra.utils import pack, unpack
from coltra.agents import Agent
from coltra.envs import SubprocVecEnv

from coltra.envs.unity_envs import UnitySimpleCrowdEnv, Mode
from coltra.envs import MultiAgentEnv

from coltra.buffers import MemoryRecord, MemoryBuffer, AgentMemoryBuffer


def collect_crowd_data(
    agent: Agent,
    env: MultiAgentEnv,
    num_steps: int,
    deterministic: bool = False,
    disable_tqdm: bool = True,
    **kwargs
) -> Tuple[MemoryRecord, Dict, Tuple]:
    """
    Performs a rollout of the agents in the environment, for an indicated number of steps or episodes.

    Args:
        agent: Agent with which to collect the data
        env: Environment in which the agent will act
        num_steps: number of steps to take; either this or num_episodes has to be passed (not both)
        deterministic: whether each agent should use the greedy policy; False by default
        disable_tqdm: whether a live progress bar should be (not) displayed

    Returns:
        data: a nested dictionary with the data
        metrics: a dictionary of metrics passed by the environment
    """
    memory = MemoryBuffer()

    # reset_start: change here in case I ever need to not reset
    obs_dict = env.reset(**kwargs)

    # state = {
    #     agent_id: self.agents[agent_id].get_initial_state(requires_grad=False) for agent_id in self.agent_ids
    # }
    metrics = {}

    for step in trange(num_steps, disable=disable_tqdm):
        # Converts a dict to a compact array which will be fed to the network - needs rethinking
        obs_array, agent_keys = pack(obs_dict)

        # Centralize the action computation for better parallelization
        actions, states, extra = agent.act(obs_array, (), deterministic, get_value=True)
        values = extra["value"]

        action_dict = unpack(
            actions, agent_keys
        )  # Convert an array to a agent-indexed dictionary
        values_dict = unpack(values, agent_keys)

        # Actual step in the environment
        next_obs, reward_dict, done_dict, info_dict = env.step(action_dict)

        all_metrics = {
            k: v
            for k, v in info_dict.items()
            if k.startswith("m_") or k.startswith("e_")
        }

        for key in all_metrics:
            metrics.setdefault(key[2:], []).append(all_metrics[key])

        memory.append(obs_dict, action_dict, reward_dict, values_dict, done_dict)

        obs_dict = next_obs

    metrics = {key: np.concatenate(value) for key, value in metrics.items()}

    # Get the last values
    obs_array, agent_keys = pack(obs_dict)

    last_values = agent.value(obs_array).detach().view(-1)

    data = memory.crowd_tensorify(last_value=last_values)

    data_shape = tuple(last_values.shape) + (num_steps,)
    return data, metrics, data_shape


def collect_renders(
    agent: Agent,
    env: MultiAgentEnv,
    num_steps: int,
    deterministic: bool = True,
    disable_tqdm: bool = False,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Performs a rollout of the agent in the environment, recording the renders

    Args:
        agent: Agent with which to collect the data
        env: Environment in which the agent will act
        num_steps: number of steps to take; either this or num_episodes has to be passed (not both)
        deterministic: whether each agent should use the greedy policy; False by default
        disable_tqdm: whether a live progress bar should be (not) displayed

    Returns:
        data: a nested dictionary with the data
        metrics: a dictionary of metrics passed by the environment
    """

    obs_dict = env.reset(**kwargs)
    agent_id = list(obs_dict.keys())[0]  # Only consider a single agent

    renders = []
    metrics = {}
    returns = []
    reward = 0

    render = env.render("rgb_array")
    renders.append(render)

    for step in trange(num_steps, disable=disable_tqdm):
        # Converts a dict to a compact array which will be fed to the network - needs rethinking
        obs_array, agent_keys = pack(obs_dict)

        # Centralize the action computation for better parallelization
        actions, states, extra = agent.act(
            obs_array, (), deterministic, get_value=False
        )

        action_dict = unpack(
            actions, agent_keys
        )  # Convert an array to a agent-indexed dictionary

        # Actual step in the environment
        next_obs, reward_dict, done_dict, info_dict = env.step(action_dict)

        render = env.render("rgb_array")
        renders.append(render)

        reward += reward_dict[agent_id]
        if done_dict[agent_id]:
            returns.append(reward)
            reward = 0

        all_metrics = {
            k: v
            for k, v in info_dict.items()
            if k.startswith("m_") or k.startswith("e_")
        }

        for key in all_metrics:
            metrics.setdefault(key[2:], []).append(all_metrics[key])

        obs_dict = next_obs

    # metrics = {key: np.concatenate(value) for key, value in metrics.items()}
    renders = np.stack(renders)
    if len(returns) > 0:
        returns = np.array(returns)
    else:
        returns = np.array([reward], dtype=np.float32)

    return renders, returns


# def collect_heterogeneous_data(
#     agent_group: MacroAgent,
#     env: MultiAgentEnv,
#     num_steps: Optional[int] = None,
#     mode: Mode = Mode.Random,
#     num_agents: Optional[int] = None,
#     deterministic: bool = False,
#     disable_tqdm: bool = True,
# ) -> Tuple[MemoryRecord, Dict]:
#     """
#     Performs a rollout of the agents in the environment, for an indicated number of steps or episodes.
#
#     Args:
#         agent_group: MacroAgent with which to collect the data
#         env: Environment in which the agent will act
#         num_steps: number of steps to take; either this or num_episodes has to be passed (not both)
#         mode: which environment should be used
#         num_agents: how many agents in the environment
#         deterministic: whether each agent should use the greedy policy; False by default
#         disable_tqdm: whether a live progress bar should be (not) displayed
#
#     Returns:
#         data: a nested dictionary with the data
#         metrics: a dictionary of metrics passed by the environment
#     """
#     memory = MemoryBuffer()
#
#     # reset_start: change here in case I ever need to not reset
#     obs_dict = env.reset(mode=mode, num_agents=num_agents)
#
#     # state = {
#     #     agent_id: self.agents[agent_id].get_initial_state(requires_grad=False) for agent_id in self.agent_ids
#     # }
#     metrics = {}
#
#     for step in trange(num_steps, disable=disable_tqdm):
#         # Compute the action for each agent
#         # action_info = {  # action, logprob, entropy, state, sm
#         #     agent_id: self.agents[agent_id].compute_single_action(obs[agent_id],
#         #                                                           # state[agent_id],
#         #                                                           deterministic[agent_id])
#         #     for agent_id in obs
#         # }
#
#         # Converts a dict to a compact array which will be fed to the network - needs rethinking
#
#         # Centralize the action computation for better parallelization
#         action_dict, states, extra = agent_group.act(
#             obs_dict, deterministic, get_value=True
#         )
#
#         values_dict = extra["value"]
#
#         # Actual step in the environment
#         next_obs, reward_dict, done_dict, info_dict = env.step(action_dict)
#         # breakpoint()
#
#         # Collect the metrics passed by the environment
#         # if isinstance(info_dict, tuple):  # SubProcVecEnv
#         #     # all_metrics = np.concatenate([info["metrics"] for info in info_dict])
#         #     all_metrics = {}
#         #     for key in info_dict[0].keys():
#         #         if key.startswith("m_"):
#         #             all_metrics[key] = np.concatenate([val[key] for val in info_dict])
#         # else:
#         #     # all_metrics = info_dict["metrics"]
#
#         all_metrics = {k: v for k, v in info_dict.items() if k.startswith("m_")}
#
#         for key in all_metrics:
#             metrics.setdefault(key[2:], []).append(all_metrics[key])
#
#         memory.append(obs_dict, action_dict, reward_dict, values_dict, done_dict)
#
#         obs_dict = next_obs
#
#     metrics = {key: np.array(value) for key, value in metrics.items()}
#
#     data = memory.crowd_tensorify()  # TODO: add heterogeneity support here for metrics?
#     return data, metrics


# def _collection_worker(agent: Agent, i: int, env_path: str, num_steps: int, base_seed: int) -> Tuple[DataBatch, Dict]:
#     # seed = round(time.time() % 100000) + i  # Ensure it's different every time
#     seed = base_seed * 100 + i
#     env = UnitySimpleCrowdEnv(file_name=env_path, no_graphics=True, worker_id=i, timeout_wait=5, seed=seed)
#     env.engine_channel.set_configuration_parameters(time_scale=100)
#     data, metrics = collect_crowd_data(agent, env, num_steps)
#     env.close()
#     return data, metrics
