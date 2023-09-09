from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from typing import (
    List,
    Dict,
    Union,
    get_type_hints,
    Callable,
    Optional,
    Any,
    TypeVar,
    Type,
    Sequence,
    Iterator,
    Tuple,
    NamedTuple,
)

import numpy as np

import torch
from torch import Tensor


class LSTMStateT(NamedTuple):
    policy_state: Tuple[Tensor, Tensor]
    value_state: Tuple[Tensor, Tensor]

    def cpu(self) -> LSTMStateT:
        return LSTMStateT(
            policy_state=(self.policy_state[0].cpu(), self.policy_state[1].cpu()),
            value_state=(self.value_state[0].cpu(), self.value_state[1].cpu()),
        )

    def cuda(self) -> LSTMStateT:
        return LSTMStateT(
            policy_state=(self.policy_state[0].cuda(), self.policy_state[1].cuda()),
            value_state=(self.value_state[0].cuda(), self.value_state[1].cuda()),
        )

    def slice(self, item: slice):
        return LSTMStateT(
            policy_state=(
                self.policy_state[0][item, ...],
                self.policy_state[1][item, ...],
            ),
            value_state=(
                self.value_state[0][item, ...],
                self.value_state[1][item, ...],
            ),
        )


Array = Union[np.ndarray, torch.Tensor]
AgentName = str
AgentNameStub = str
PolicyName = str
LSTMState = (
    tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]]
    | tuple[tuple[Tensor, Tensor], tuple]
    | tuple
    | LSTMStateT
)


def get_batch_size(tensor: Union[Tensor, Multitype, LSTMStateT]) -> int:
    if isinstance(tensor, Tensor):
        _tensor: Tensor = tensor  # Just for types
        return _tensor.shape[0]
    elif isinstance(tensor, Multitype):
        _multitensor: Multitype = tensor
        return _multitensor.batch_size
    elif isinstance(tensor, LSTMStateT):
        _lstm_state: LSTMStateT = tensor
        return _lstm_state.policy_state[0].shape[0] if _lstm_state != () else -1
    elif isinstance(tensor, tuple):  # Also LSTM state?
        _tuple: tuple = tensor
        return _tuple[0].shape[1] if _tuple != () else -1


def is_array(x: Any) -> bool:
    return isinstance(x, np.ndarray) or isinstance(x, torch.Tensor)


class Multitype:
    _dict: dict[str, Array]

    @classmethod
    def stack_tensor(cls, value_list: Sequence[Multitype], dim: int = 0):
        res = cls()
        keys = value_list[0]._dict.keys()  # assume all the inputs have the same keys
        for key in keys:
            stacked = np.stack([value[key] for value in value_list], axis=dim)
            value = torch.as_tensor(stacked)

            # tensors = [torch.as_tensor(value[key]) for value in value_list]

            # value = torch.stack(tensors, dim=dim)
            res._dict[key] = value

        return res

    @classmethod
    def cat_tensor(cls, value_list: Sequence[Multitype], dim: int = 0):
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

    def map(self, func: Callable[[Array], Array]):
        """Applies a function to each field, returns a new object"""
        res = type(self)()
        for key in self._dict.keys():
            value = self._dict[key]
            new_field = func(value)
            res._dict[key] = new_field
        return res

    def cuda(self, *args, **kwargs):
        return self.map(lambda x: x.cuda(*args, **kwargs))

    def cpu(self):
        return self.map(lambda x: x.cpu())

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

T = TypeVar("T", np.ndarray, torch.Tensor, Multitype, tuple)


def concat(array_list: list[T], dim: int = 0) -> T:
    """
    Concatenates a list of arrays or multitypes into a single array.

    Args:
        array_list: list of arrays or multitypes
        dim: dimension to concatenate along

    Returns:
        array or multitype
    """
    if len(array_list) == 0:
        raise ValueError("Cannot concatenate an empty list")

    arr = array_list[0]

    if isinstance(arr, Multitype):
        return type(arr).cat_tensor(array_list, dim=dim)
    elif isinstance(arr, np.ndarray):
        return np.concatenate(array_list, axis=dim)
    elif isinstance(arr, torch.Tensor):
        return torch.cat(array_list, dim=dim)
    elif isinstance(arr, tuple):
        concatenated_states = LSTMStateT(
            *(
                tuple(
                    torch.cat(
                        [state_tuple[i][j] for state_tuple in array_list], dim=dim
                    )
                    for j in range(len(arr[0]))
                )
                for i in range(len(arr))
            )
        )
        return concatenated_states


R = TypeVar("R")  # Should be a subclass of Record


@dataclass
class Record:
    def apply(self, func: Callable[[Array], Array]):
        """Applies a function to each field, returns a new object"""
        kwargs = {}
        for field_ in fields(self):
            value = getattr(self, field_.name)
            new_field = func(value) if value is not None else None
            kwargs[field_.name] = new_field
        res = OnPolicyRecord(**kwargs)
        return res

    def cuda(self, *args, **kwargs):
        return self.apply(lambda x: x.cuda(*args, **kwargs))

    def cpu(self):
        return self.apply(lambda x: x.cpu())

    @classmethod
    def crowdify(cls, memory_dict: dict[str, R]) -> R:
        """
        Converts a dictionary of memory records to a single memory record.
        """
        tensor_data = memory_dict.values()
        return OnPolicyRecord(
            **{
                field_.name: concat(
                    [getattr(agent_buffer, field_.name) for agent_buffer in tensor_data]
                )
                for field_ in fields(cls)
            }
        )


@dataclass
class AgentBuffer:
    def append(self, record):
        for field_ in fields(record):
            name = field_.name
            record_value = getattr(record, name)
            if record_value is not None:
                getattr(self, name).append(record_value)


@dataclass
class OnPolicyRecord(Record):
    obs: Observation
    action: Action
    reward: Reward
    value: Value
    done: Done
    last_value: Optional[Value]
    state: Optional[LSTMStateT] = None


@dataclass
class AgentOnPolicyBuffer(AgentBuffer):
    obs: list[Observation] = field(default_factory=list)
    action: list[Action] = field(default_factory=list)
    reward: list[Reward] = field(default_factory=list)
    value: list[Value] = field(default_factory=list)
    done: list[Done] = field(default_factory=list)
    state: list[Optional[tuple]] = field(default_factory=list)


@dataclass
class OnPolicyBuffer:
    data: dict[str, AgentOnPolicyBuffer] = field(default_factory=dict)

    def append(
        self,
        obs: dict[str, Observation],
        action: dict[str, Action],
        reward: dict[str, Reward],
        value: dict[str, Value],
        done: dict[str, Done],
        state: Optional[dict[str, tuple]] = None,
    ):

        for agent_id in obs:  # Assume the keys are identical
            record = OnPolicyRecord(
                obs[agent_id],
                action[agent_id],
                reward[agent_id],
                value[agent_id],
                done[agent_id],
                None,
                state[agent_id] if state and state[agent_id] else None,
            )

            self.data.setdefault(agent_id, AgentOnPolicyBuffer()).append(record)

    def tensorify(
        self, data: Optional[dict[str, AgentOnPolicyBuffer]] = None
    ) -> dict[str, OnPolicyRecord]:
        if data is None:
            data = self.data
        result = {}
        for agent_id, agent_buffer in data.items():  # str -> AgentMemoryBuffer
            state_dict = {}
            if agent_buffer.state:
                for i, state in enumerate(agent_buffer.state):
                    state_dict[f"agent_{i}"] = state

                packed_state, _ = pack_lstm_states(state_dict)
            else:
                packed_state = None

            result[agent_id] = OnPolicyRecord(
                obs=Observation.stack_tensor(agent_buffer.obs),
                action=Action.stack_tensor(agent_buffer.action),
                reward=torch.as_tensor(agent_buffer.reward),
                value=torch.as_tensor(agent_buffer.value),
                done=torch.as_tensor(agent_buffer.done),
                last_value=None,
                state=packed_state,
            )
        return result

    def crowd_tensorify(
        self,
        data: Optional[dict[str, AgentOnPolicyBuffer]] = None,
        last_value: Optional[Value] = None,
    ) -> OnPolicyRecord:
        if data is None:
            data = self.data
        tensor_data = self.tensorify(data).values()
        all_states = [
            agent_buffer.state
            for agent_buffer in tensor_data
            if agent_buffer.state is not None
        ]

        # Packing LSTM states if any are present
        packed_state = None
        if all_states:
            state_dict = {f"agent_{i}": state for i, state in enumerate(all_states)}
            packed_state, _ = pack_lstm_states(state_dict)

        return OnPolicyRecord(
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
            state=packed_state,  # New packed state
        )

    def hetero_tensorify(
        self,
        data: Optional[dict[AgentName, AgentOnPolicyBuffer]] = None,
        last_value: Optional[dict[AgentName, Value]] = None,
        policy_mapping: Optional[dict[AgentNameStub, PolicyName]] = None,
    ) -> dict[PolicyName, OnPolicyRecord]:
        if data is None:
            data = self.data

        if policy_mapping is None:
            policy_mapping = {"": "crowd"}

        if last_value is None:
            last_value = {}  # TODO: fix this? But I probably won't use it ever

        policy_rev = {v: k for k, v in policy_mapping.items()}
        policies = set(policy_mapping.values())

        result = {}
        for policy in policies:

            policy_last_value, _ = pack_tensor(
                {
                    k: v
                    for k, v in last_value.items()
                    if k.startswith(policy_rev[policy])
                }
            )

            result[policy] = self.crowd_tensorify(
                data={
                    k: v for k, v in data.items() if k.startswith(policy_rev[policy])
                },
                last_value=policy_last_value,
            )

        return result

    # def family_tensorify(self, last_value: Optional[dict[str, Value]] = None) -> tuple[OnPolicyRecord, OnPolicyRecord]:
    #     tensor_data = self.tensorify()
    #     family_data, crowd_data = split_dict(tensor_data)
    #     family_last_value, crowd_last_value = split_dict(last_value)
    #
    #     return (
    #         OnPolicyRecord(
    #             obs=Observation.cat_tensor(
    #                 [agent_buffer.obs for agent_buffer in family_data.values()]
    #             ),
    #             action=Action.cat_tensor(
    #                 [agent_buffer.action for agent_buffer in family_data.values()]
    #             ),
    #             reward=torch.cat([agent_buffer.reward for agent_buffer in family_data.values()]),
    #             value=torch.cat([agent_buffer.value for agent_buffer in family_data.values()]),
    #             done=torch.cat([agent_buffer.done for agent_buffer in family_data.values()]),
    #             last_value=last_value,
    #         ),
    #         OnPolicyRecord(
    #             obs=Observation.cat_tensor(
    #                 [agent_buffer.obs for agent_buffer in crowd_data.values()]
    #             ),
    #             action=Action.cat_tensor(
    #                 [agent_buffer.action for agent_buffer in crowd_data.values()]
    #             ),
    #             reward=torch.cat([agent_buffer.reward for agent_buffer in crowd_data.values()]),
    #             value=torch.cat([agent_buffer.value for agent_buffer in crowd_data.values()]),
    #             done=torch.cat([agent_buffer.done for agent_buffer in crowd_data.values()]),
    #             last_value=last_value,
    #         ),
    #     )


@dataclass
class DQNRecord(Record):
    obs: Observation
    action: Action
    reward: Reward
    next_obs: Observation
    done: Done


@dataclass
class AgentDQNBuffer(AgentBuffer):
    maxlen: int = 100000
    obs: deque[Observation] = field(default_factory=deque)
    action: deque[Action] = field(default_factory=deque)
    reward: deque[Reward] = field(default_factory=deque)
    next_obs: deque[Observation] = field(default_factory=deque)
    done: deque[Done] = field(default_factory=deque)

    def __post_init__(self):
        self.obs = deque(maxlen=self.maxlen)
        self.action = deque(maxlen=self.maxlen)
        self.reward = deque(maxlen=self.maxlen)
        self.next_obs = deque(maxlen=self.maxlen)
        self.done = deque(maxlen=self.maxlen)


@dataclass
class DQNBuffer:  # TODO: this and OnPolicyBuffer should have a common base class?
    maxlen: int = 100000
    data: dict[str, AgentDQNBuffer] = field(default_factory=dict)

    def append(
        self,
        obs: dict[str, Observation],
        action: dict[str, Action],
        reward: dict[str, Reward],
        next_obs: dict[str, Observation],
        done: dict[str, Done],
    ):
        for agent_id in obs:  # Assume the keys are identical
            record = DQNRecord(
                obs[agent_id],
                action[agent_id],
                reward[agent_id],
                next_obs[agent_id],
                done[agent_id],
            )

            self.data.setdefault(agent_id, AgentDQNBuffer(maxlen=self.maxlen)).append(
                record
            )

    def tensorify(self) -> dict[str, DQNRecord]:
        result = {}
        for agent_id, agent_buffer in self.data.items():  # str -> AgentMemoryBuffer
            result[agent_id] = DQNRecord(
                obs=Observation.stack_tensor(agent_buffer.obs),
                action=Action.stack_tensor(agent_buffer.action),
                reward=torch.as_tensor(agent_buffer.reward),
                next_obs=Observation.stack_tensor(agent_buffer.next_obs),
                done=torch.as_tensor(agent_buffer.done),
            )
        return result

    def crowd_tensorify(self) -> DQNRecord:
        tensor_data = self.tensorify().values()
        return DQNRecord(
            obs=Observation.cat_tensor(
                [agent_buffer.obs for agent_buffer in tensor_data]
            ),
            action=Action.cat_tensor(
                [agent_buffer.action for agent_buffer in tensor_data]
            ),
            reward=torch.cat([agent_buffer.reward for agent_buffer in tensor_data]),
            next_obs=Observation.cat_tensor(
                [agent_buffer.next_obs for agent_buffer in tensor_data]
            ),
            done=torch.cat([agent_buffer.done for agent_buffer in tensor_data]),
        )


class TensorDict(Mapping):
    def __init__(self, data: Array, names: Sequence[str]):
        super().__init__()
        self.data = data
        self.names = names
        self.inverse_names = {name: i for i, name in enumerate(names)}

        assert len(self.names) == len(self.data)

    def __getitem__(self, k: str) -> Array:
        return self.data[self.inverse_names[k]]

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self) -> Iterator[str]:
        return iter(self.names)


K = TypeVar("K", bound=str)
V = TypeVar("V")


def split_dict(dict_batch: dict[K, V]) -> tuple[dict[K, V], dict[K, V]]:
    family_dict = {k: v for k, v in dict_batch.items() if "Family" in k}
    crowd_dict = {k: v for k, v in dict_batch.items() if "Person" in k}

    return family_dict, crowd_dict


def pack(dict_: dict[str, Observation]) -> Tuple[Observation, List[str]]:
    keys = list(dict_.keys())
    values = Observation.stack_tensor([dict_[key] for key in keys])

    return values, keys


def pack_tensor(dict_: dict[str, Array]) -> Tuple[Tensor, List[str]]:
    keys = list(dict_.keys())
    if isinstance(dict_[keys[0]], torch.Tensor):
        values = torch.stack([dict_[key] for key in keys])
    else:
        values = torch.as_tensor(np.stack([dict_[key] for key in keys]))

    return values, keys


def unpack(arrays: Any, keys: List[str]) -> dict[str, Any]:
    value_dict = {key: arrays[i] for i, key in enumerate(keys)}
    return value_dict


def pack_lstm_states(state_dict: Dict[str, LSTMState]) -> Tuple[LSTMState, List[str]]:
    keys = list(state_dict.keys())
    states = [state_dict[key] for key in keys]

    state_empty = len(states[0]) == 0 or len(states[0][0]) == 0
    if state_empty:
        return (), []
    value_state_empty = len(states[0][1]) == 0

    # Policy states
    policy_hiddens = torch.cat([state[0][0] for state in states], dim=0)
    policy_cells = torch.cat([state[0][1] for state in states], dim=0)

    packed_state = (policy_hiddens, policy_cells)

    # Only pack value states if they are not empty
    if not value_state_empty:
        value_hiddens = torch.cat([state[1][0] for state in states], dim=0)
        value_cells = torch.cat([state[1][1] for state in states], dim=0)
        value_state = (value_hiddens, value_cells)
    else:
        value_state = ()

    packed_state = (packed_state, value_state)

    return packed_state, keys


def unpack_lstm_states(state: LSTMState, keys: List[str]) -> Dict[str, LSTMState]:
    policy_hiddens, policy_cells = state[0]
    num_states = len(keys)

    # Unpack policy states
    policy_hiddens_split = torch.split(policy_hiddens, [1] * num_states, dim=0)
    policy_cells_split = torch.split(policy_cells, [1] * num_states, dim=0)

    # Initialize as empty tuples
    value_hiddens_split = ()
    value_cells_split = ()

    # Unpack value state if it is not empty
    if state[1]:
        value_hiddens, value_cells = state[1]
        value_hiddens_split = torch.split(value_hiddens, [1] * num_states, dim=0)
        value_cells_split = torch.split(value_cells, [1] * num_states, dim=0)

    unpacked_states = {
        keys[i]: (
            (policy_hiddens_split[i], policy_cells_split[i]),
            (value_hiddens_split[i], value_cells_split[i])
            if value_hiddens_split
            else (),
        )
        for i in range(num_states)
    }

    return unpacked_states
