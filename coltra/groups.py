import abc
import os
from functools import reduce
from itertools import chain
from typing import Dict, Callable, Tuple, Iterable, Any, Optional
import numpy as np
from typing import List, TypeVar

import torch.nn
from torch import Tensor
from coltra.utils import augment_observations
from coltra.agents import Agent
from coltra.buffers import (
    Observation,
    Action,
    split_dict,
    pack,
    unpack,
    pack_lstm_states,
    unpack_lstm_states,
    LSTMState,
)

AgentName = str
AgentNameStub = str
PolicyName = str


class PolicyNameError(Exception):
    pass


class MacroAgent(abc.ABC):
    policy_mapping: dict[AgentNameStub, PolicyName]

    def get_policy_name(self, agent_name: AgentName) -> PolicyName:
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
        obs_dict: dict[AgentName, Observation],
        deterministic: bool = False,
        get_value: bool = False,
    ) -> Tuple[dict[AgentName, Action], Tuple, dict]:
        pass

    @abc.abstractmethod
    def evaluate(
        self,
        obs_batch: dict[PolicyName, Observation],
        action_batch: dict[PolicyName, Action],
    ) -> dict[PolicyName, Tuple[Tensor, Tensor, Tensor]]:
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
    def value(
        self,
        obs_batch: dict[AgentName, Observation],
        action_batch: dict[AgentName, Action],
        **kwargs,
    ) -> dict[AgentName, Tensor]:
        pass

    @abc.abstractmethod
    def save(self, *args, **kwargs):
        pass


class HomogeneousGroup(MacroAgent):
    """
    A simple macroagent with a single policy
    """

    def __init__(self, agent: Agent, policy_name: str = "crowd"):
        self.policy_name = policy_name
        self.policy_mapping = {"": self.policy_name}
        self.agent = agent
        self.agents = {self.policy_name: agent}
        self.policies = [self.policy_name]

    def act(
        self,
        obs_dict: dict[AgentName, Observation],
        deterministic: bool = False,
        get_value: bool = False,
        state_dict: Optional[dict[AgentName, tuple]] = None,
    ):
        if len(obs_dict) == 0:
            return {}, {}, {}
        obs, keys = pack(obs_dict)

        states, s_keys = pack_lstm_states(state_dict)

        actions, states, extra = self.agent.act(
            obs_batch=obs,
            state_batch=states,
            deterministic=deterministic,
            get_value=get_value,
        )

        actions_dict = unpack(actions, keys)
        if states:
            new_states_dict = unpack_lstm_states(states, s_keys)
        else:
            new_states_dict = state_dict

        extra = {key: unpack(value, keys) for key, value in extra.items()}

        return actions_dict, new_states_dict, extra

    def parameters(self) -> Iterable[torch.nn.Parameter]:
        return self.agent.model.parameters()

    def cuda(self):
        self.agent.cuda()

    def cpu(self):
        self.agent.cpu()

    def value(
        self,
        obs_batch: dict[AgentName, Observation],
        state_dict: dict[AgentName, LSTMState],
        **kwargs,
    ) -> dict[str, Tensor]:
        obs, keys = pack(obs_batch)
        states, s_keys = pack_lstm_states(state_dict)
        values = self.agent.value(obs, states)
        return unpack(values, keys)

    def value_pack(
        self,
        obs_batch: dict[AgentName, Observation],
        state_dict: dict[AgentName, LSTMState],
        **kwargs,
    ) -> Tensor:
        obs, _ = pack(obs_batch)
        states, _ = pack_lstm_states(state_dict)

        values, _ = self.agent.value(obs, state_batch=states)
        return values

    T = TypeVar("T")

    def embed(self, value: T) -> dict[PolicyName, T]:
        return {self.policy_name: value}

    def evaluate(
        self,
        obs_batch: dict[PolicyName, Observation],
        action_batch: dict[PolicyName, Action],
        state: Optional[dict[PolicyName, tuple]] = None,
    ) -> dict[PolicyName, Tuple[Tensor, Tensor, Tensor]]:

        obs = obs_batch[self.policy_name]
        action = action_batch[self.policy_name]
        state = state[self.policy_name] if state else ()
        return self.embed(self.agent.evaluate(obs, action, state))

    def embed_evaluate(
        self, obs: Observation, action: Action, state: tuple = ()
    ) -> Tuple[Tensor, Tensor, Tensor]:
        return self.evaluate(self.embed(obs), self.embed(action), self.embed(state))[
            self.policy_name
        ]

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

    def get_initial_state(
        self, obs_batch: dict[AgentName, Observation], requires_grad: bool = False
    ) -> dict[AgentName, tuple]:
        return {
            agent_name: self.agent.get_initial_state(requires_grad=requires_grad)
            for agent_name in obs_batch.keys()
        }


class FamilyGroup(MacroAgent):
    """
    A macroagent with a family agent whose actions get added to the crowd agent's observations

    GENERAL OUTLINE:
    Compute family actions
    Append them to the crowd observations (separate function?)
    Pack the crowd observations
    Compute crowd actions
    Unpack crowd actions
    Append family actions to crowd actions
    Return all actions

    Probably need to properly handle extras (values? what else was there?)

    Also do the same for evaluate - should be simpler since it's PolicyName and not AgentName?


    TODO:
    - evaluate
    - save/load
    - trainer
    - optimizer - do I need a new one? or just fix parameters?
    """

    def __init__(self, agent: Agent, family_agent: Agent):
        self.policy_name = "crowd"
        # self.policy_mapping = {"": self.policy_name}
        self.policy_mapping = {"Person": "crowd", "Family": "family"}
        self.agent = agent
        self.family_agent = family_agent

        self.agents = {"crowd": agent, "family": family_agent}
        self.policies = ["crowd", "family"]

    def act(
        self,
        obs_dict: dict[AgentName, Observation],
        deterministic: bool = False,
        get_value: bool = False,
    ):
        if len(obs_dict) == 0:
            return {}, (), {}

        family_dict, crowd_dict = split_dict(obs_dict)

        family_obs, family_keys = pack(family_dict)

        family_actions, _, family_extra = self.family_agent.act(
            obs_batch=family_obs,
            state_batch=(),
            deterministic=deterministic,
            get_value=get_value,
        )

        family_actions_dict = unpack(family_actions, family_keys)

        augment_observations(crowd_dict, family_actions_dict)

        crowd_obs, crowd_keys = pack(crowd_dict)
        actions, states, extra = self.agent.act(
            obs_batch=crowd_obs,
            state_batch=(),
            deterministic=deterministic,
            get_value=get_value,
        )

        actions_dict = unpack(actions, crowd_keys)

        actions_dict.update(family_actions_dict)

        extra = {key: unpack(value, crowd_keys) for key, value in extra.items()}
        family_extra = {
            key: unpack(value, family_keys) for key, value in family_extra.items()
        }

        for key, value in family_extra.items():
            if key in extra:
                extra[key].update(value)
            else:
                extra[key] = value

        extra["augmented_obs"] = {**crowd_dict, **family_dict}

        return actions_dict, states, extra

    def evaluate(
        self,
        obs_batch: dict[PolicyName, Observation],
        action_batch: dict[PolicyName, Action],
    ) -> dict[PolicyName, Tuple[Tensor, Tensor, Tensor]]:
        # Assumption: observations are already augmented

        return {
            policy: self.agents[policy].evaluate(
                obs_batch[policy], action_batch[policy]
            )
            for policy in self.policies
        }

    def parameters(self) -> Iterable[torch.nn.Parameter]:
        return self.agent.model.parameters()

    def cuda(self):
        self.agent.cuda()
        self.family_agent.cuda()

    def cpu(self):
        self.agent.cpu()
        self.family_agent.cpu()

    def value(
        self,
        obs_batch: dict[AgentName, Observation],
        action_batch: dict[AgentName, Action],
        **kwargs,
    ) -> dict[str, Tensor]:
        family_obs, crowd_obs = split_dict(obs_batch)
        family_actions, crowd_actions = split_dict(action_batch)

        family_obs, family_keys = pack(family_obs)
        family_values = self.family_agent.value(family_obs, ())

        augment_observations(crowd_obs, family_actions)

        crowd_obs, crowd_keys = pack(crowd_obs)

        crowd_values = self.agent.value(crowd_obs, ())

        crowd_values = unpack(crowd_values, crowd_keys)
        family_values = unpack(family_values, family_keys)

        values = {**crowd_values, **family_values}

        return values

    # def value_pack(
    #     self,
    #     obs_batch: dict[AgentName, Observation],
    #     action_batch: dict[AgentName, Action],
    #     **kwargs,
    # ) -> Tensor:
    #     obs, _ = pack(obs_batch)
    #     values = self.agent.value(obs)
    #     return values

    def save(
        self,
        base_path: str,
    ):
        agent_fname: str = "agent.pt"
        model_fname: str = "model.pt"
        family_fname: str = "family_agent.pt"
        family_model_fname: str = "family_model.pt"

        mapping_fname: str = "policy_mapping.pt"

        torch.save(self.agent, os.path.join(base_path, agent_fname))
        torch.save(self.agent.model, os.path.join(base_path, model_fname))
        torch.save(self.family_agent, os.path.join(base_path, family_fname))
        torch.save(self.family_agent.model, os.path.join(base_path, family_model_fname))
        torch.save(self.policy_mapping, os.path.join(base_path, mapping_fname))

    @classmethod
    def load(cls, base_path: str, weight_idx: Optional[int] = None):
        agent_fname: str = "agent.pt"
        mapping_fname: str = "policy_mapping.pt"

        weight_fname: str = "weights"

        family_fname: str = "family_agent.pt"
        family_model_fname: str = "family_model.pt"

        device = None if torch.cuda.is_available() else "cpu"
        agent = torch.load(os.path.join(base_path, agent_fname), map_location=device)
        family_agent = torch.load(
            os.path.join(base_path, family_fname), map_location=device
        )
        group = cls(agent, family_agent)

        if weight_idx == -1:
            weight_idx = max(
                [
                    int(fname.split("_")[-1])  # Get the last agent
                    for fname in os.listdir(os.path.join(base_path, "saved_weights"))
                    if fname.startswith(weight_fname)
                ]
            )

        if not torch.cuda.is_available():
            group.cpu()

        if weight_idx is not None:
            group.load_state(base_path=base_path, idx=weight_idx)

        return group

    def save_state(self, base_path: str, idx: int):
        weights_dir = os.path.join(base_path, "saved_weights")
        if not os.path.exists(weights_dir):
            os.mkdir(weights_dir)

        torch.save(
            self.agent.model.state_dict(),
            os.path.join(weights_dir, f"weights_{idx}"),
        )

        torch.save(
            self.family_agent.model.state_dict(),
            os.path.join(weights_dir, f"family_weights_{idx}"),
        )

    def load_state(self, base_path: str, idx: int = -1):
        weights_path = os.path.join(base_path, "saved_weights", f"weights_{idx}")
        weights = torch.load(weights_path, map_location=self.agent.model.device)

        family_weights_path = os.path.join(
            base_path, "saved_weights", f"family_weights_{idx}"
        )
        family_weights = torch.load(
            family_weights_path, map_location=self.family_agent.model.device
        )

        self.agent.model.load_state_dict(weights)
        self.family_agent.model.load_state_dict(family_weights)


# class HeterogeneousGroup(MacroAgent):
#     """
#     A "macroagent" combining several individual agents
#     """
#
#     def __init__(
#             self,
#             agents: dict[PolicyName, Agent],
#             policy_mapping: Optional[dict[AgentNameStub, PolicyName]] = None,
#     ):
#         super().__init__()
#         self.agents: dict[PolicyName, HomogeneousGroup] = {key: HomogeneousGroup(agent)
#                                                            for key, agent in agents.items()}
#         self.policy_mapping = policy_mapping or {"": "agent"}
#
#         self.policy_names = list(set(self.policy_mapping.values()))
#
#     def act(
#             self,
#             obs_dict: dict[str, Observation],
#             deterministic: bool = False,
#             get_value: bool = False,
#     ):
#
#         policy_obs: dict[PolicyName, dict[AgentName, Observation]] = {}
#
#         for agent_id, obs in obs_dict:
#             policy_name = self.get_policy_name(agent_id)
#             policy_obs.setdefault(policy_name, {})[agent_id] = obs
#
#         actions = {}
#         part_extras = {}  # TODO: rename this
#         extras = {}
#         for policy_name, policy_obs_dict in policy_obs:
#             agent = self.agents[policy_name]
#             policy_actions, states, extra = agent.act(policy_obs_dict, deterministic, get_value)
#             actions = {**actions, **policy_actions}
#             for extra_name, extra_dict in extra.items():
#                 part_extras.setdefault(extra_name, []).append(extra_dict)
#
#             for key, dicts in part_extras.items():
#                 extras[key] = reduce(lambda acc, x: x | acc, dicts)
#
#         return actions, (), extras
#
#     def evaluate(self, obs_batch: dict[PolicyName, Observation],
#                  action_batch: dict[PolicyName, Action]) -> dict[PolicyName, Tuple[Tensor, Tensor, Tensor]]:
#         pass
#
#     def parameters(self) -> Iterable[torch.nn.Parameter]:
#         policy_params = [agent.parameters() for agent in self.agents.values()]
#         return chain.from_iterable(policy_params)
#
#     def cuda(self):
#         for agent in self.agents.values():
#             agent.cuda()
#
#     def cpu(self):
#         for agent in self.agents.values():
#             agent.cpu()
#
#     def value(self, obs_batch: dict[AgentName, Observation], **kwargs) -> dict[AgentName, Tensor]:
#         policy_obs: dict[PolicyName, dict[AgentName, Observation]] = {}
#
#         for agent_id, obs in obs_batch:
#             policy_name = self.get_policy_name(agent_id)
#             policy_obs.setdefault(policy_name, {})[agent_id] = obs
#
#         values = {}
#         for policy_name, policy_obs_dict in policy_obs:
#             agent = self.agents[policy_name]
#             policy_values = agent.value(policy_obs_dict)
#             values = {**values, **policy_values}
#
#         return values
#
#     def save(self, *args, **kwargs):
#         pass
