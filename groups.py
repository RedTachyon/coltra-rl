import abc
from typing import Dict, Callable, Tuple
import numpy as np
from typing import List

from coltra.utils import pack, unpack
from coltra.agents import Agent
from coltra.buffers import Observation, Action


class MacroAgent(abc.ABC):
    policy_names: List[str]

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.policy_names = []

    @abc.abstractmethod
    def act(self,
            obs_dict: Dict[str, Observation],
            deterministic: bool = False,
            get_value: bool = False) -> Tuple[Dict[str, Action], Tuple, Dict]:
        pass


class HomogeneousGroup(MacroAgent):
    """
    A simple macroagent with a single policy
    """
    def __init__(self, agent: Agent):
        super().__init__()
        self.policy_names.append("crowd")
        self.agent = agent

    def act(self,
            obs_dict: Dict[str, Observation],
            deterministic: bool = False,
            get_value: bool = False):

        obs, keys = pack(obs_dict)
        actions, states, extra = self.agent.act(obs, (), deterministic, get_value=get_value)

        actions_dict = unpack(actions, keys)

        extra = {key: unpack(value, keys) for key, value in extra.items() }

        return actions_dict, states, extra

    # def evaluate(self, obs_batch: Observation, action_batch: Action) -> Tuple[Tensor, Tensor, Tensor]:
    #     raise NotImplementedError


class HeterogeneousGroup(MacroAgent):
    """
    A "macroagent" combining several individual agents
    """
    def __init__(self, agents: Dict[str, Agent], policy_mapper: Callable[[str], str] = lambda x: x):
        super().__init__()
        self.agents = {key: HomogeneousGroup(agent) for key, agent in agents.items()}
        self.policy_mapper = policy_mapper

    def act(self,
            obs_dict: Dict[str, Observation],
            deterministic: bool = False,
            get_value: bool = False):

        policy_obs = {}

        for key, obs in obs_dict:
            policy_name = self.policy_mapper(key)
            policy_obs.setdefault(policy_name, {})[key] = obs

        assert set(policy_obs.keys()).issubset(self.agents.keys())
        # TODO: Reconsider variable names here

        all_actions = {}
        all_extras = {}
        states = ()  # TODO: fix for recurrent policies

        for policy_name, obs in policy_obs:
            agent = self.agents[policy_name]
            actions_dict, states, extra = agent.act(obs, deterministic, get_value)

            all_actions.update(actions_dict)

            for key, extra_dict in extra:
                all_extras.setdefault(key, {}).update(extra_dict)

        return all_actions, states, all_extras

