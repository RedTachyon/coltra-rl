from typing import Tuple, Optional, Any, Callable

import numpy as np
from tqdm import trange

from coltra.buffers import OnPolicyRecord, OnPolicyBuffer
from coltra.envs.base_env import MultiAgentEnv
from coltra.groups import HomogeneousGroup, FamilyGroup, MacroAgent

from coltra.utils import attention_string


def collect_crowd_data(
    agents: HomogeneousGroup,
    env: MultiAgentEnv,
    num_steps: int,
    deterministic: bool = False,
    disable_tqdm: bool = True,
    env_kwargs: Optional[dict[str, Any]] = None,
) -> Tuple[dict[str, OnPolicyRecord], dict, Tuple]:
    """
    Performs a rollout of the agents in the environment, for an indicated number of steps or episodes.

    Args:
        agents: Agent with which to collect the data
        env: Environment in which the agent will act
        num_steps: number of steps to take; either this or num_episodes has to be passed (not both)
        deterministic: whether each agent should use the greedy policy; False by default
        disable_tqdm: whether a live progress bar should be (not) displayed
        env_kwargs: arguments to pass to the environment

    Returns:
        data: a nested dictionary with the data
        metrics: a dictionary of metrics passed by the environment
    """
    if env_kwargs is None:
        env_kwargs = {}
    memory = OnPolicyBuffer()

    # reset_start: change here in case I ever need to not reset
    obs_dict = env.reset(**env_kwargs)
    states = agents.get_initial_state(obs_dict, requires_grad=False)

    metrics = {}

    for step in trange(num_steps, disable=disable_tqdm):
        # Converts a dict to a compact array which will be fed to the network - needs rethinking
        # obs_array, agent_keys = pack(obs_dict)

        # Centralize the action computation for better parallelization
        action_dict, new_states, extra = agents.act(
            obs_dict, deterministic=deterministic, get_value=True, state_dict=states
        )
        values_dict = extra["value"]

        # action_dict = unpack(
        #     actions, agent_keys
        # )  # Convert an array to a agent-indexed dictionary
        # values_dict = unpack(values, agent_keys)

        # Actual step in the environment
        next_obs, reward_dict, done_dict, info_dict = env.step(action_dict)

        all_metrics = {
            k: v
            for k, v in info_dict.items()
            if k.startswith("m_") or k.startswith("e_")
        }

        for key in all_metrics:
            metrics.setdefault(key, []).append(all_metrics[key])

        memory.append(
            obs_dict, action_dict, reward_dict, values_dict, done_dict, state=states
        )

        obs_dict = next_obs
        states = new_states

    metrics = {key: np.concatenate(value) for key, value in metrics.items()}

    last_values = agents.value_pack(obs_dict, state_dict=states).detach().view(-1)

    # Get the last values
    # obs_array, agent_keys = pack(obs_dict)

    # last_values = agents.value(obs_array).detach().view(-1)

    data = memory.crowd_tensorify(last_value=last_values)

    data_shape = tuple(last_values.shape) + (num_steps,)
    return agents.embed(data), metrics, data_shape


def collect_family_data(
    agents: FamilyGroup,
    env: MultiAgentEnv,
    num_steps: int,
    deterministic: bool = False,
    disable_tqdm: bool = True,
    env_kwargs: Optional[dict[str, Any]] = None,
) -> Tuple[dict[str, OnPolicyRecord], dict, dict[str, tuple]]:
    """
    Performs a rollout of the agents in the environment, for an indicated number of steps or episodes.

    Args:
        agents: Agent with which to collect the data
        env: Environment in which the agent will act
        num_steps: number of steps to take; either this or num_episodes has to be passed (not both)
        deterministic: whether each agent should use the greedy policy; False by default
        disable_tqdm: whether a live progress bar should be (not) displayed
        env_kwargs: arguments to pass to the environment

    Returns:
        data: a nested dictionary with the data
        metrics: a dictionary of metrics passed by the environment
    """
    if env_kwargs is None:
        env_kwargs = {}
    memory = OnPolicyBuffer()

    # reset_start: change here in case I ever need to not reset
    obs_dict = env.reset(**env_kwargs)

    # state = {
    #     agent_id: self.agents[agent_id].get_initial_state(requires_grad=False) for agent_id in self.agent_ids
    # }
    metrics = {}

    for step in trange(num_steps, disable=disable_tqdm):
        # Converts a dict to a compact array which will be fed to the network - needs rethinking
        # obs_array, agent_keys = pack(obs_dict)

        # Centralize the action computation for better parallelization
        action_dict, states, extra = agents.act(obs_dict, deterministic, get_value=True)
        values_dict = extra["value"]
        obs_dict = extra["augmented_obs"]

        # action_dict = unpack(
        #     actions, agent_keys
        # )  # Convert an array to a agent-indexed dictionary
        # values_dict = unpack(values, agent_keys)

        # Actual step in the environment
        next_obs, reward_dict, done_dict, info_dict = env.step(action_dict)

        all_metrics = {
            k: v
            for k, v in info_dict.items()
            if k.startswith("m_") or k.startswith("e_")
        }

        for key in all_metrics:
            metrics.setdefault(key, []).append(all_metrics[key])

        memory.append(obs_dict, action_dict, reward_dict, values_dict, done_dict)

        obs_dict = next_obs

    metrics = {key: np.concatenate(value) for key, value in metrics.items()}

    _, _, extra = agents.act(obs_dict, deterministic, get_value=True)
    last_values = extra["value"]
    # last_values = agents.value_pack(obs_dict).detach().view(-1)

    # Get the last values
    # obs_array, agent_keys = pack(obs_dict)

    # last_values = agents.value(obs_array).detach().view(-1)

    data = memory.hetero_tensorify(
        last_value=last_values, policy_mapping=agents.policy_mapping
    )

    data_shapes = {
        k: (v.reward.shape[0] // num_steps, num_steps) for k, v in data.items()
    }
    # data_shape = tuple(last_values.shape) + (num_steps,)
    return data, metrics, data_shapes


def collect_renders(
    agents: MacroAgent,
    env: MultiAgentEnv,
    num_steps: int,
    deterministic: bool = True,
    disable_tqdm: bool = False,
    env_kwargs: Optional[dict] = None,
    show_attention: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Performs a rollout of the agent in the environment, recording the renders

    Args:
        agents: Agent group with which to collect the data
        env: Environment in which the agent will act
        num_steps: number of steps to take; either this or num_episodes has to be passed (not both)
        deterministic: whether each agent should use the greedy policy; False by default
        disable_tqdm: whether a live progress bar should be (not) displayed

    Returns:
        data: a nested dictionary with the data
        metrics: a dictionary of metrics passed by the environment
    """

    if env_kwargs is None:
        env_kwargs = {}
    obs_dict = env.reset(**env_kwargs)
    agent_id = list(obs_dict.keys())[0]  # Only consider a single agent

    renders = []
    metrics = {}
    returns = []
    reward = 0

    render = env.render(mode="rgb_array")
    renders.append(render)

    for step in trange(num_steps, disable=disable_tqdm):
        # Converts a dict to a compact array which will be fed to the network - needs rethinking

        action_dict, states, extra = agents.act(
            obs_dict, deterministic=deterministic, get_value=False
        )

        if show_attention:
            from coltra.envs.unity_envs import UnitySimpleCrowdEnv

            attention = extra["attention"]
            attention_msg = attention_string(attention)
            assert isinstance(
                env, UnitySimpleCrowdEnv
            ), "Can't visualize attention outside of Unity crowd envs"

            env.attention_channel.send_string(attention_msg)

        # Actual step in the environment
        next_obs, reward_dict, done_dict, info_dict = env.step(action_dict)

        render = env.render(mode="rgb_array")
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
