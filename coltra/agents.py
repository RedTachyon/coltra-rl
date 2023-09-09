import os
from typing import Tuple, Optional, Dict
import abc

import numpy as np

import torch
from torch import Tensor
from torch.distributions import Normal, Categorical
import gymnasium as gym

from coltra.models.base_models import BaseModel
from coltra.buffers import Observation, Action
from coltra.models.model_utils import ContCategorical


class Agent:
    model: BaseModel
    device: str

    def __init__(self, *args, **kwargs):
        pass

    def act(
        self,
        obs_batch: Observation,
        state_batch: Tuple = (),
        deterministic: bool = False,
        get_value: bool = False,
        **kwargs,
    ) -> Tuple[Action, Tuple, dict]:
        """Return: Action, State, Extras"""
        raise NotImplementedError

    def evaluate(
        self, obs_batch: Observation, action_batch: Action, state_batch: tuple = ()
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Return: logprobs, values, entropies"""
        raise NotImplementedError

    def value(
        self, obs_batch: Observation, state_batch: tuple, **kwargs
    ) -> tuple[Tensor, tuple]:
        raise NotImplementedError

    def cuda(self):
        if self.model is not None:
            self.device = "cuda"
            self.model.cuda()

    def cpu(self):
        if self.model is not None:
            self.device = "cpu"
            self.model.cpu()

    def state_dict(self, *args, **kwargs):
        return self.model.state_dict(*args, **kwargs)

    def get_initial_state(self, batch_size: int = 1, requires_grad=True):
        return getattr(self.model, "get_initial_state", lambda *x, **xx: ())(
            batch_size=batch_size, requires_grad=requires_grad
        )

    def save(
        self,
        base_path: str,
        agent_fname: str = "agent.pt",
        model_fname: str = "model.pt",
    ):
        torch.save(self, os.path.join(base_path, agent_fname))
        torch.save(self.model, os.path.join(base_path, model_fname))

    @staticmethod
    def load(
        base_path: str,
        weight_idx: Optional[int] = None,
        agent_fname: str = "agent.pt",
        weight_fname: str = "weights",
    ):
        device = None if torch.cuda.is_available() else "cpu"
        agent = torch.load(os.path.join(base_path, agent_fname), map_location=device)

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

            agent.model.load_state_dict(weights)

        if not torch.cuda.is_available():
            agent.cpu()

        return agent

    @property
    def unwrapped(self):
        return self

    def __setstate__(self, state):
        for key in state:
            setattr(self, key, state[key])

    def __getstate__(self):
        return self.__dict__


class CAgent(Agent):  # Continuous Agent
    model: BaseModel

    def __init__(self, model: BaseModel, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.stateful = model.stateful

    def act(
        self,
        obs_batch: Observation,  # [B, ...]
        state_batch: tuple = (),
        deterministic: bool = False,
        get_value: bool = False,
        **kwargs,
    ) -> Tuple[Action, tuple, dict]:
        """Computes the action for an observation,
        passes along the state for recurrent models, and optionally the value"""
        obs_batch = obs_batch.tensor(self.model.device)
        state_batch = tuple(
            tuple(tensor.to(self.model.device) for tensor in state)
            for state in state_batch
        )

        action_distribution: Normal
        states: tuple
        actions: Tensor

        with torch.no_grad():
            action_distribution, states, extra_outputs = self.model(
                obs_batch, state_batch, get_value=get_value
            )

            if deterministic:
                actions = action_distribution.loc
            else:
                actions = action_distribution.rsample()

        # extra = {}
        if get_value:
            value = extra_outputs["value"]
            extra_outputs["value"] = value.squeeze(-1).cpu().numpy()

        return Action(continuous=actions.cpu().numpy()), states, extra_outputs

    def evaluate(
        self, obs_batch: Observation, action_batch: Action, state_batch: tuple = ()
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Computes action logprobs, observation values and policy entropy for each of the (obs, action, hidden_state)
        transitions. Preserves all the necessary gradients.

        Args:
            obs_batch: observations collected with the collector
            action_batch: actions taken by the agent

        Returns:
            action_logprobs: tensor of action logprobs (batch_size, )
            values: tensor of observation values (batch_size, )
            entropies: tensor of entropy values (batch_size, )
        """
        obs_batch = obs_batch.tensor(self.model.device)
        action_batch = action_batch.tensor(self.model.device)
        state_batch = tuple(
            tuple(tensor.to(self.model.device) for tensor in state)
            for state in state_batch
        )

        action_distribution, _, extra_outputs = self.model(
            obs_batch, state=state_batch, get_value=True
        )
        values = extra_outputs["value"].sum(-1)
        # Sum across dimensions of the action
        action_logprobs = action_distribution.log_prob(action_batch.continuous).sum(-1)
        entropies = action_distribution.entropy().sum(-1)

        return action_logprobs, values, entropies

    def value(
        self, obs_batch: Observation, state_batch: tuple = (), **kwargs
    ) -> tuple[Tensor, tuple]:
        obs_batch = obs_batch.tensor(self.model.device)
        state_batch = tuple(
            tuple(tensor.to(self.model.device) for tensor in state)
            for state in state_batch
        )

        values, state_batch = self.model.value(
            obs_batch.tensor(self.model.device), state_batch[1] if state_batch else ()
        )
        return values, state_batch


class DAgent(Agent):
    model: BaseModel

    def __init__(self, model: BaseModel, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.stateful = model.stateful

    def act(
        self,
        obs_batch: Observation,
        state_batch: Tuple = (),
        deterministic: bool = False,
        get_value: bool = False,
        **kwargs,
    ) -> Tuple[Action, Tuple, dict]:

        obs_batch = obs_batch.tensor(self.model.device)
        state_batch = tuple(
            tuple(tensor.to(self.model.device) for tensor in state)
            for state in state_batch
        )

        action_distribution: Categorical
        states: Tuple
        actions: Tensor

        with torch.no_grad():
            action_distribution, states, extra_outputs = self.model(
                obs_batch, state_batch, get_value=get_value
            )

            if deterministic:
                actions = action_distribution.probs.argmax(dim=-1)
            else:
                actions = action_distribution.sample()

        # extra = {}
        if get_value:
            value = extra_outputs["value"]
            extra_outputs["value"] = value.squeeze(-1).cpu().numpy()

        return Action(discrete=actions.cpu().numpy()), states, extra_outputs

    def evaluate(
        self,
        obs_batch: Observation,
        action_batch: Action,
        state_batch: tuple = (),
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Computes action logprobs, observation values and policy entropy for each of the (obs, action)
        transitions. Works on discrete actions.

        Args:
            obs_batch: observations collected with the collector
            action_batch: actions taken by the agent

        Returns:
            action_logprobs: tensor of action logprobs (batch_size, )
            values: tensor of observation values (batch_size, )
            entropies: tensor of entropy values (batch_size, )
        """
        obs_batch = obs_batch.tensor(self.model.device)
        action_batch = action_batch.tensor(self.model.device)
        state_batch = tuple(
            tuple(tensor.to(self.model.device) for tensor in state)
            for state in state_batch
        )

        action_distribution, _, extra_outputs = self.model(
            obs_batch, state=state_batch, get_value=True
        )
        values = extra_outputs["value"].sum(-1)
        # Sum across dimensions of the action
        action_logprobs = action_distribution.log_prob(action_batch.discrete)
        entropies = action_distribution.entropy()

        return action_logprobs, values, entropies

    def value(
        self, obs_batch: Observation, state_batch: tuple = (), **kwargs
    ) -> tuple[Tensor, tuple]:
        obs_batch = obs_batch.tensor(self.model.device)
        state_batch = tuple(
            tuple(tensor.to(self.model.device) for tensor in state)
            for state in state_batch
        )

        values, state_batch = self.model.value(
            obs_batch.tensor(self.model.device), state_batch[1] if state_batch else ()
        )
        return values, state_batch


class MixedAgent(Agent):
    model: BaseModel

    def __init__(self, model: BaseModel, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.stateful = model.stateful

    def act(
        self,
        obs_batch: Observation,
        state_batch: Tuple = (),
        deterministic: bool = False,
        get_value: bool = False,
        **kwargs,
    ) -> Tuple[Action, Tuple, dict]:

        obs_batch = obs_batch.tensor(self.model.device)
        state_batch = tuple(s.to(self.model.device) for s in state_batch)

        action_distribution: ContCategorical
        states: Tuple
        actions: Tensor

        with torch.no_grad():
            action_distribution, states, extra_outputs = self.model(
                obs_batch, state_batch, get_value=get_value
            )

            if deterministic:
                actions = action_distribution.deterministic_sample()
            else:
                actions = action_distribution.sample()

        # extra = {}
        if get_value:
            value = extra_outputs["value"]
            extra_outputs["value"] = value.squeeze(-1).cpu().numpy()

        return (
            Action(
                discrete=actions[..., 0].cpu().numpy(),
                continuous=actions[..., 1:].cpu().numpy(),
            ),
            states,
            extra_outputs,
        )

    def evaluate(
        self, obs_batch: Observation, action_batch: Action
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Computes action logprobs, observation values and policy entropy for each of the (obs, action)
        transitions. Works on discrete actions.

        Args:
            obs_batch: observations collected with the collector
            action_batch: actions taken by the agent

        Returns:
            action_logprobs: tensor of action logprobs (batch_size, )
            values: tensor of observation values (batch_size, )
            entropies: tensor of entropy values (batch_size, )
        """
        obs_batch = obs_batch.tensor(self.model.device)
        action_batch = action_batch.tensor(self.model.device)
        # state_batch = data_batch['states']

        action_distribution, _, extra_outputs = self.model(obs_batch, get_value=True)
        values = extra_outputs["value"].sum(-1)
        # Sum across dimensions of the action
        action_logprobs = action_distribution.log_prob(action_batch)
        entropies = action_distribution.entropy()

        return action_logprobs, values, entropies

    def value(self, obs_batch: Observation, **kwargs) -> tuple[Tensor, tuple]:
        obs_batch = obs_batch.tensor(self.model.device)
        values, _ = self.model.value(obs_batch.tensor(self.model.device), ())
        return values, ()


class ToyAgent(Agent):
    model = None

    def act(
        self,
        obs_batch: Observation,
        state_batch: Tuple = (),
        deterministic: bool = False,
        get_value: bool = False,
        **kwargs,
    ) -> Tuple[Action, Tuple, dict]:
        """Return: Action, State, Extras"""
        raise NotImplementedError

    def evaluate(
        self, obs_batch: Observation, action_batch: Action
    ) -> Tuple[Tensor, Tensor, Tensor]:
        zero = torch.zeros((obs_batch.batch_size,))
        return zero, zero, zero

    def value(self, obs_batch: Observation, **kwargs) -> tuple[Tensor, tuple]:
        zero = torch.zeros((obs_batch.batch_size, 1))
        return zero, ()


class RandomGymAgent(ToyAgent):
    def __init__(self, action_space: gym.Space, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_space = action_space

    def act(
        self,
        obs_batch: Observation,
        state_batch: Tuple = (),
        deterministic: bool = False,
        get_value: bool = False,
        **kwargs,
    ) -> Tuple[Action, Tuple, dict]:
        batch_size = obs_batch.batch_size

        if isinstance(self.action_space, gym.spaces.Box):
            _action = np.tile(self.action_space.sample(), (batch_size, 1))
            # if batch_size == 1:
            #     _action = _action.ravel()
            action = Action(continuous=_action)
        else:
            action = Action(
                discrete=np.array(
                    [self.action_space.sample() for _ in range(batch_size)]
                )
            )

        return (
            action,
            (),
            {"value": np.zeros((batch_size,))},
        )


class ConstantAgent(ToyAgent):
    def __init__(self, action: np.ndarray, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action = np.asarray(action, dtype=np.float32)

    def act(
        self,
        obs_batch: Observation,
        state_batch: Tuple = (),
        deterministic: bool = False,
        get_value: bool = False,
        **kwargs,
    ) -> Tuple[Action, Tuple, dict]:
        batch_size = obs_batch.batch_size

        return (
            Action(continuous=np.tile(self.action, (batch_size, 1))),
            (),
            {"value": np.zeros((batch_size,))},
        )


class RandomDAgent(ToyAgent):
    def __init__(self, num_actions: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_actions = num_actions

    def act(
        self,
        obs_batch: Observation,
        state_batch: Tuple = (),
        deterministic: bool = False,
        get_value: bool = False,
        **kwargs,
    ) -> Tuple[Action, Tuple, dict]:
        batch_size = obs_batch.batch_size

        return (
            Action(discrete=np.random.randint(0, self.num_actions, batch_size)),
            (),
            {"value": np.zeros((batch_size,))},
        )


def AutoAgent(model: BaseModel) -> Agent:
    if model.discrete:
        return DAgent(model)
    else:
        return CAgent(model)
