from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import List, Dict, Union, get_type_hints, Callable, Optional, Any

import numpy as np

import torch
from torch import Tensor

Array = Union[np.ndarray, torch.Tensor]


def get_batch_size(tensor: Union[Tensor, Multitype]) -> int:
    if isinstance(tensor, Tensor):
        _tensor: Tensor = tensor  # Just for types
        return _tensor.shape[0]
    else:
        _multitensor: Multitype = tensor
        return _multitensor.batch_size


def is_array(x: Any) -> bool:
    return isinstance(x, np.ndarray) or isinstance(x, torch.Tensor)


class Multitype:
    _dict: Dict[str, Array]

    @classmethod
    def stack_tensor(cls, value_list: list[Multitype], dim: int = 0):
        res = cls()
        keys = value_list[0]._dict.keys()  # assume all the inputs have the same keys
        for key in keys:
            tensors = [torch.as_tensor(value[key]) for value in value_list]

            value = torch.stack(tensors, dim=dim)
            res._dict[key] = value

        return res

    @classmethod
    def cat_tensor(cls, value_list: list[Multitype], dim: int = 0):
        res = cls()
        keys = value_list[0]._dict.keys()  # assume all the inputs have the same keys
        for key in keys:
            tensors = [torch.as_tensor(value[key]) for value in value_list]

            value = torch.cat(tensors, dim=dim)
            res._dict[key] = value

        return res

    @property
    def batch_size(self) -> int:
        value = -1
        for key in self._dict.keys():
            field_value = self._dict[key]
            # TODO: Fix this
            if field_value is None:
                continue
            _batch_size = (
                field_value.shape[0]
                if (len(field_value.shape) > 1 or key == "discrete")
                else 1
            )
            if value < 0:
                value = _batch_size
            elif value >= 0:
                assert (
                    value == field_value.shape[0]
                ), "Different types have different batch sizes"

        return value

    def tensor(self, device: str = "cpu"):
        res = type(self)()
        for key in self._dict.keys():
            value = self._dict[key]
            tensor = torch.as_tensor(value).to(device)
            res._dict[key] = tensor
        return res

    def apply(self, func: Callable[[Array], Array]):
        """Applies a function to each field, returns a new object"""
        res = type(self)()
        for key in self._dict.keys():
            value = self._dict[key]
            new_field = func(value)
            res._dict[key] = new_field
        return res

    def cuda(self, *args, **kwargs):
        return self.apply(lambda x: x.cuda(*args, **kwargs))

    def cpu(self):
        return self.apply(lambda x: x.cpu())

    def __getitem__(self, item: Union[str, int, slice]) -> Union[Array, Multitype]:
        if isinstance(item, str):
            return self._dict[item]
        else:
            res = type(self)()
            for key in self._dict.keys():
                value = self._dict[key]
                new_field = value[item]
                res._dict[key] = new_field
            return res

    def __getattr__(self, item) -> Array:
        if item not in self._dict:
            raise AttributeError(
                f"Attribute {item} not found in this instance of {type(self).__name__}"
            )
        return self._dict[item]

    def __repr__(self):
        inside_str = ", ".join([f"{key}={value}" for key, value in self._dict.items()])
        return f"{type(self).__name__}({inside_str})"

    def __getstate__(self):
        return self._dict

    def __setstate__(self, state):
        self._dict = state


BaseObs = Union[Array, dict[str, Array]]


class Observation(Multitype):
    def __init__(self, obs: Optional[BaseObs] = None, **kwargs: Array):
        if obs is None:
            self._dict = {}
        elif obs is not None and is_array(obs):
            self._dict = {"vector": obs}
        else:  # is not None and not is_array => is a dict
            self._dict = obs

        self._dict = {**self._dict, **kwargs}


BaseAction = Union[Array, int, dict[str, Array]]


class Action(Multitype):
    def __init__(self, action: Optional[BaseAction] = None, **kwargs: Array):
        if action is None:
            self._dict = {}
        elif action is not None and is_array(action):  # default is continuous action
            self._dict = {"continuous": action}
        else:  # is not None and not is_array => is a dict
            self._dict = action

        self._dict = {**self._dict, **kwargs}


def discrete(value: Array) -> Action:
    return Action(discrete=value)


def continuous(value: Array) -> Action:
    return Action(continuous=value)


Reward = Array  # float32
LogProb = Array  # float32
Value = Array  # float32
Done = Union[Array, bool]  # bool


@dataclass
class MemoryRecord:
    obs: Observation
    action: Action
    reward: Reward
    value: Value
    done: Done
    last_value: Optional[Value]

    def apply(self, func: Callable[[Array], Array]):
        """Applies a function to each field, returns a new object"""
        kwargs = {}
        for field_ in fields(self):
            value = getattr(self, field_.name)
            new_field = func(value) if value is not None else None
            kwargs[field_.name] = new_field
        res = MemoryRecord(**kwargs)
        return res

    def cuda(self, *args, **kwargs):
        return self.apply(lambda x: x.cuda(*args, **kwargs))

    def cpu(self):
        return self.apply(lambda x: x.cpu())

    @classmethod
    def crowdify(cls, memory_dict: Dict[str, MemoryRecord]) -> MemoryRecord:
        tensor_data = memory_dict.values()
        return MemoryRecord(
            obs=Observation.cat_tensor(
                [agent_buffer.obs for agent_buffer in tensor_data]
            ),
            action=Action.cat_tensor(
                [agent_buffer.action for agent_buffer in tensor_data]
            ),
            reward=torch.cat([agent_buffer.reward for agent_buffer in tensor_data]),
            value=torch.cat([agent_buffer.value for agent_buffer in tensor_data]),
            done=torch.cat([agent_buffer.done for agent_buffer in tensor_data]),
            last_value=torch.cat(
                [agent_buffer.last_value for agent_buffer in tensor_data]
            ),
        )


@dataclass
class AgentMemoryBuffer:
    obs: List[Observation] = field(default_factory=list)
    action: List[Action] = field(default_factory=list)
    reward: List[Reward] = field(default_factory=list)
    value: List[Value] = field(default_factory=list)
    done: List[Done] = field(default_factory=list)

    def append(self, record):
        for field_ in fields(record):
            name = field_.name
            record_value = getattr(record, name)
            if record_value is not None:
                getattr(self, name).append(record_value)


@dataclass
class MemoryBuffer:
    data: Dict[str, AgentMemoryBuffer] = field(default_factory=dict)

    def append(
        self,
        obs: Dict[str, Observation],
        action: Dict[str, Action],
        reward: Dict[str, Reward],
        value: Dict[str, Value],
        done: Dict[str, Done],
    ):

        for agent_id in obs:  # Assume the keys are identical
            record = MemoryRecord(
                obs[agent_id],
                action[agent_id],
                reward[agent_id],
                value[agent_id],
                done[agent_id],
                None,
            )

            self.data.setdefault(agent_id, AgentMemoryBuffer()).append(record)

    # def append(self, *args):
    #
    #     for agent_id in args[0]:  # Assume the keys are identical
    #         record = MemoryRecord(*args)
    #
    #         self.data.setdefault(agent_id, AgentMemoryBuffer()).append(record)

    def tensorify(self) -> Dict[str, MemoryRecord]:
        result = {}
        for agent_id, agent_buffer in self.data.items():  # str -> AgentMemoryBuffer
            result[agent_id] = MemoryRecord(
                obs=Observation.stack_tensor(agent_buffer.obs),
                action=Action.stack_tensor(agent_buffer.action),
                reward=torch.tensor(agent_buffer.reward),
                value=torch.tensor(agent_buffer.value),
                done=torch.tensor(agent_buffer.done),
                last_value=None,
            )
        return result

    def crowd_tensorify(self, last_value: Optional[Value] = None) -> MemoryRecord:
        tensor_data = self.tensorify().values()
        return MemoryRecord(
            obs=Observation.cat_tensor(
                [agent_buffer.obs for agent_buffer in tensor_data]
            ),
            action=Action.cat_tensor(
                [agent_buffer.action for agent_buffer in tensor_data]
            ),
            reward=torch.cat([agent_buffer.reward for agent_buffer in tensor_data]),
            value=torch.cat([agent_buffer.value for agent_buffer in tensor_data]),
            done=torch.cat([agent_buffer.done for agent_buffer in tensor_data]),
            last_value=last_value,
        )

    # def crowd_tensorify(self, last_value: Optional[Value] = None) -> MemoryRecord:
    #     tensor_data = self.tensorify().values()
    #     return MemoryRecord(
    #         obs=Observation.stack_tensor(
    #             [agent_buffer.obs for agent_buffer in tensor_data]
    #         ),
    #         action=Action.stack_tensor(
    #             [agent_buffer.action for agent_buffer in tensor_data]
    #         ),
    #         reward=torch.stack([agent_buffer.reward for agent_buffer in tensor_data]),
    #         value=torch.stack([agent_buffer.value for agent_buffer in tensor_data]),
    #         done=torch.stack([agent_buffer.done for agent_buffer in tensor_data]),
    #         last_value=last_value,
    #     )
