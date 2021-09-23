from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import List, Dict, Union, get_type_hints, Callable, Optional

import numpy as np

import torch
from torch import Tensor

TensorArray = Union[np.ndarray, torch.Tensor]


def get_batch_size(tensor: Union[Tensor, Multitype]) -> int:
    if isinstance(tensor, Tensor):
        _tensor: Tensor = tensor  # Just for types
        return _tensor.shape[0]
    else:
        _multitensor: Multitype = tensor
        return _multitensor.batch_size


@dataclass
class Multitype:
    @classmethod
    def stack_tensor(cls, value_list: List[Multitype], dim: int = 0):
        res = cls()
        for field_ in get_type_hints(cls):
            tensors = [
                torch.as_tensor(getattr(value, field_))
                for value in value_list
                if getattr(value, field_) is not None
            ]

            value = torch.stack(tensors, dim=dim) if tensors else None
            setattr(res, field_, value)

        return res

    @classmethod
    def cat_tensor(cls, value_list: List[Multitype], dim: int = 0):
        res = cls()
        for field_ in get_type_hints(cls):
            tensors = [
                torch.as_tensor(getattr(value, field_))
                for value in value_list
                if getattr(value, field_) is not None
            ]

            value = torch.cat(tensors, dim=dim) if tensors else None
            setattr(res, field_, value)

        return res

    @property
    def batch_size(self) -> int:
        value = -1
        for field_ in fields(self):
            field_value = getattr(self, field_.name)
            # TODO: Fix this shit
            if field_value is None:
                continue
            _batch_size = field_value.shape[0] if (len(field_value.shape) > 1 or field_.name == "discrete") else 1
            if value < 0:
                value = _batch_size
            elif value >= 0:
                assert (
                    value == field_value.shape[0]
                ), "Different types have different batch sizes"

        return value

    def tensor(self, device: str = "cpu"):
        res = type(self)()
        for field_ in fields(self):
            value = getattr(self, field_.name)
            tensor = torch.as_tensor(value).to(device) if value is not None else None
            setattr(res, field_.name, tensor)
        return res

    def __getitem__(self, item):
        res = type(self)()
        for field_ in fields(self):
            value = getattr(self, field_.name)
            part = value[item] if value is not None else None
            setattr(res, field_.name, part)
        return res

    def apply(self, func: Callable[[TensorArray], TensorArray]):
        """Applies a function to each field, returns a new object"""
        res = type(self)()
        for field_ in fields(self):
            value = getattr(self, field_.name)
            new_field = func(value) if value is not None else None
            setattr(res, field_.name, new_field)
        return res

    def cuda(self, *args, **kwargs):
        return self.apply(lambda x: x.cuda(*args, **kwargs))

    def cpu(self):
        return self.apply(lambda x: x.cpu())


@dataclass
class Observation(Multitype):
    vector: Optional[TensorArray] = None
    rays: Optional[TensorArray] = None
    buffer: Optional[TensorArray] = None
    image: Optional[TensorArray] = None


@dataclass
class Action(Multitype):
    continuous: Optional[TensorArray] = None
    discrete: Optional[TensorArray] = None


Reward = TensorArray  # float32
LogProb = TensorArray  # float32
Value = TensorArray  # float32
Done = Union[TensorArray, bool]  # bool


@dataclass
class MemoryRecord:
    obs: Observation
    action: Action
    reward: Reward
    value: Value
    done: Done

    def apply(self, func: Callable[[TensorArray], TensorArray]):
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
            )
        return result

    # TODO: reconsider whether different agents' experiences should be concatenated or stacked?
    def crowd_tensorify(self) -> MemoryRecord:
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
        )
    # def crowd_tensorify(self) -> MemoryRecord:
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
    #     )