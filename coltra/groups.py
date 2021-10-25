import abc
from typing import Dict, Callable, Tuple, Iterable, Any, Optional
import numpy as np
from typing import List, TypeVar

import torch.nn
from torch import Tensor
from coltra.utils import pack, unpack
from coltra.agents import Agent
from coltra.buffers import Observation, Action


class PolicyNameError(Exception):
    pass


class MacroAgent(abc.ABC):
    policy_mapping: Dict[str, str]

    def get_policy_name(self, agent_name: str) -> str:
        policy_mapping = {
            key: value
            for key, value in sorted(
                self.policy_mapping.items(), key=lambda x: len(x[1]), reverse=True
            )
        }
        for key, value in policy_mapping:
            if agent_name.startswith(key):
                return value

        raise PolicyNameError(
            f"Cannot match {agent_name} to any of the policy names: {list(policy_mapping)}"
        )

    @abc.abstractmethod
    def act(
        self,
        obs_dict: Dict[str, Observation],
        deterministic: bool = False,
        get_value: bool = False,
    ) -> Tuple[Dict[str, Action], Tuple, Dict]:
        pass

    @abc.abstractmethod
    def evaluate(
        self, obs_batch: Observation, action_batch: Action
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Return: logprobs, values, entropies"""
        pass

    @abc.abstractmethod
    def parameters(self) -> Iterable[torch.nn.Parameter]:
        pass

    @abc.abstractmethod
    def cuda(self):
        pass

    @abc.abstractmethod
    def cpu(self):
        pass

    @abc.abstractmethod
    def value(self, obs_batch: Dict[str, Observation], **kwargs) -> Dict[str, Tensor]:
        pass

    # TODO: add save
    # @abc.abstractmethod
    # def save(self, *args, **kwargs):
    #     pass


class HomogeneousGroup(MacroAgent):
    """
    A simple macroagent with a single policy
    """

    def __init__(self, agent: Agent):
        self.policy_name = "crowd"
        self.policy_mapping = {"": self.policy_name}
        self.agent = agent

    def act(
        self,
        obs_dict: Dict[str, Observation],
        deterministic: bool = False,
        get_value: bool = False,
    ):
        obs, keys = pack(obs_dict)
        actions, states, extra = self.agent.act(
            obs, (), deterministic, get_value=get_value
        )

        actions_dict = unpack(actions, keys)

        extra = {key: unpack(value, keys) for key, value in extra.items()}

        return actions_dict, states, extra

    def parameters(self) -> Iterable[torch.nn.Parameter]:
        return self.agent.model.parameters()

    def cuda(self):
        self.agent.cuda()

    def cpu(self):
        self.agent.cpu()

    def value(self, obs_batch: Dict[str, Observation], **kwargs) -> Dict[str, Tensor]:
        obs, keys = pack(obs_batch)
        values = self.agent.value(obs)
        return unpack(values, keys)

    def value_pack(self, obs_batch: Dict[str, Observation], **kwargs) -> Tensor:
        obs, _ = pack(obs_batch)
        values = self.agent.value(obs)
        return values

    T = TypeVar("T")

    def embed(self, value: T) -> Dict[str, T]:
        return {self.policy_name: value}

    def evaluate(
        self, obs_batch: Dict[str, Observation], action_batch: Dict[str, Action]
    ) -> Dict[str, Tuple[Tensor, Tensor, Tensor]]:

        obs = obs_batch[self.policy_name]
        action = action_batch[self.policy_name]
        return self.embed(self.agent.evaluate(obs, action))

    def embed_evaluate(self, obs: Observation, action: Action) -> Tuple[Tensor, Tensor, Tensor]:
        return self.evaluate(self.embed(obs), self.embed(action))[self.policy_name]


# class HeterogeneousGroup(MacroAgent):
#     """
#     A "macroagent" combining several individual agents
#     """
#
#     def __init__(
#         self,
#         agents: Dict[str, Agent],
#         policy_mapper: Callable[[str], str] = lambda x: x,
#     ):
#         super().__init__()
#         self.agents = {key: HomogeneousGroup(agent) for key, agent in agents.items()}
#         self.policy_mapper = policy_mapper
#
#     def act(
#         self,
#         obs_dict: Dict[str, Observation],
#         deterministic: bool = False,
#         get_value: bool = False,
#     ):
#
#         policy_obs: Dict[str, Observation] = {}
#
#         for key, obs in obs_dict:
#             policy_name = self.policy_mapper(key)
#             policy_obs.setdefault(policy_name, {})[key] = obs
#
#         assert set(policy_obs.keys()).issubset(self.agents.keys())
#         # TODO: Reconsider variable names here
#
#         all_actions = {}
#         all_extras = {}
#         states = ()  # TODO: fix for recurrent policies
#
#         for policy_name, obs in policy_obs:
#             agent = self.agents[policy_name]
#             actions_dict, states, extra = agent.act(obs, deterministic, get_value)
#
#             all_actions.update(actions_dict)
#
#             for key, extra_dict in extra:
#                 all_extras.setdefault(key, {}).update(extra_dict)
#
#         return all_actions, states, all_extras
