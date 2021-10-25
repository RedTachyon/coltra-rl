import abc
import os
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
        for key, value in policy_mapping.items():
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

    @abc.abstractmethod
    def save(self, *args, **kwargs):
        pass


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

    def embed_evaluate(
        self, obs: Observation, action: Action
    ) -> Tuple[Tensor, Tensor, Tensor]:
        return self.evaluate(self.embed(obs), self.embed(action))[self.policy_name]

    def save(
        self,
        base_path: str,
    ):
        agent_fname: str = "agent.pt"
        model_fname: str = "model.pt"
        mapping_fname: str = "policy_mapping.pt"

        torch.save(self.agent, os.path.join(base_path, agent_fname))
        torch.save(self.agent.model, os.path.join(base_path, model_fname))
        torch.save(self.policy_mapping, os.path.join(base_path, mapping_fname))

    @classmethod
    def load(cls, base_path: str, weight_idx: Optional[int] = None):
        agent_fname: str = "agent.pt"
        mapping_fname: str = "policy_mapping.pt"

        weight_fname: str = "weights"

        device = None if torch.cuda.is_available() else "cpu"
        agent = torch.load(os.path.join(base_path, agent_fname), map_location=device)
        group = cls(agent)

        if weight_idx == -1:
            weight_idx = max(
                [
                    int(fname.split("_")[-1])  # Get the last agent
                    for fname in os.listdir(os.path.join(base_path, "saved_weights"))
                    if fname.startswith(weight_fname)
                ]
            )

        if weight_idx is not None:
            weights = torch.load(
                os.path.join(
                    base_path, "saved_weights", f"{weight_fname}_{weight_idx}"
                ),
                map_location=device,
            )

            group.agent.model.load_state_dict(weights)

        if not torch.cuda.is_available():
            group.cpu()

        return group

    def save_state(self, base_path: str, idx: int):
        weights_dir = os.path.join(base_path, "saved_weights")
        if not os.path.exists(weights_dir):
            os.mkdir(weights_dir)

        torch.save(
            self.agent.model.state_dict(),
            os.path.join(weights_dir, f"weights_{idx}"),
        )

    def load_state(self, base_path: str, idx: int = -1):
        weights_path = os.path.join(base_path, "saved_weights", f"weights_{idx}")
        weights = torch.load(weights_path, map_location=self.agent.model.device)
        self.agent.model.load_state_dict(weights)



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
