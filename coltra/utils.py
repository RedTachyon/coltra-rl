from typing import Dict, List, Union, Tuple, Any, Callable, Optional, Iterator, Type

import numpy as np

import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import time

from torch.optim.optimizer import Optimizer
from torch.optim.adam import Adam
from torch.optim.adadelta import Adadelta
from torch.optim.adagrad import Adagrad
from torch.optim.adamw import AdamW
from torch.optim.adamax import Adamax
from torch.optim.sgd import SGD

from collections import defaultdict

from torch.utils.tensorboard import SummaryWriter

from coltra.buffers import Observation

DataBatch = DataBatchT = Dict[str, Dict[str, Any]]
AgentDataBatch = Dict[str, Union[Tensor, Tuple]]
Array = Union[Tensor, np.ndarray]


def write_dict(metrics: Dict[str, Union[int, float]],
               step: int,
               writer: Optional[SummaryWriter] = None):
    """Writes a dictionary to a tensorboard SummaryWriter"""
    if writer is not None:
        writer: SummaryWriter
        for key, value in metrics.items():
            writer.add_scalar(tag=key, scalar_value=value, global_step=step)


def np_float(x: float) -> np.ndarray:
    """Convenience function to create a one-element float32 numpy array"""
    return np.array([x], dtype=np.float32)


def get_optimizer(opt_name: str) -> Callable[..., Optimizer]:
    """Gets an optimizer by name"""
    optimizers = {
        "adam": Adam,
        "adadelta": Adadelta,
        "adamw": AdamW,
        "adagrad": Adagrad,
        "adamax": Adamax,
        "sgd": SGD
    }

    if opt_name not in optimizers.keys():
        raise ValueError(f"Wrong optimizer: {opt_name} is not a valid optimizer name. ")

    return optimizers[opt_name]


def get_activation(act_name: str) -> Callable[[Tensor], Tensor]:
    """Gets an activation function by name"""
    activations = {
        "relu": F.relu,
        "relu6": F.relu6,
        "elu": F.elu,
        "leaky_relu": F.leaky_relu,
        "sigmoid": F.sigmoid,
        "tanh": torch.tanh,
        "softmax": F.softmax,
        "gelu": lambda x: x * F.sigmoid(1.702 * x)
    }

    if act_name not in activations.keys():
        raise ValueError(f"Wrong activation: {act_name} is not a valid activation function name.")

    return activations[act_name]


class GELU(nn.Module):
    def __init__(self):
        super(GELU, self).__init__()

    def forward(self, x: Tensor):
        return x * F.sigmoid(1.702 * x)


def get_activation_module(act_name: str) -> Type[nn.Module]:
    """Gets an activation module by name"""
    activations: Dict[str, Type[nn.Module]] = {
        "relu": nn.ReLU,
        "relu6": nn.ReLU6,
        "elu": nn.ELU,
        "leaky_relu": nn.LeakyReLU,
        "sigmoid": nn.Sigmoid,
        "tanh": nn.Tanh,
        "softmax": nn.Softmax,
        "gelu": GELU,
    }

    if act_name not in activations.keys():
        raise ValueError(f"Wrong activation: {act_name} is not a valid activation function name.")

    return activations[act_name]


def get_initializer(init_name: str) -> Callable[[Tensor], None]:
    """Gets an initializer by name"""
    initializers = {
        "kaiming_normal": nn.init.kaiming_normal_,
        "kaiming_uniform": nn.init.kaiming_uniform_,
        "xavier_normal": nn.init.xavier_normal_,
        "xavier_uniform": nn.init.xavier_uniform_,
        "zeros": nn.init.zeros_
    }

    if init_name not in initializers.keys():
        raise ValueError(f"Wrong initialization: {init_name} is not a valid initializer name.")

    return initializers[init_name]


class Timer:
    """
    Simple timer to get temporal metrics. Upon calling .checkpoint(), returns the time since the last call
    """

    def __init__(self):
        self.start = time.time()

    def checkpoint(self) -> float:
        now = time.time()
        diff = now - self.start
        self.start = now
        return diff


def transpose_batch(data_batch: Union[DataBatch, DataBatchT]) -> Union[DataBatchT, DataBatch]:
    """
    In a 2-nested dictionary, swap the key levels. So it turns
    {
        "observations": {"Agent0": ..., "Agent1": ...},
        "actions": {"Agent0": ..., "Agent1": ...},
        ...
    }
    into
    {
        "Agent0": {"observations": ..., "actions": ..., ...},
        "Agent1": {"observations": ..., "actions": ..., ...},
    }
    Also works the other way around.
    Doesn't copy the underlying data, so it's very efficient (~30μs)
    """
    d = defaultdict(dict)
    for key1, inner in data_batch.items():
        for key2, value in inner.items():
            d[key2][key1] = value
    return dict(d)


class Masked:
    @staticmethod
    def mean(input_: Tensor, mask: Tensor) -> Tensor:
        """Mean of elements not covered by the mask"""
        return torch.sum(input_ * mask) / torch.sum(mask)

    @staticmethod
    def accuracy(preds: Tensor, labels: Tensor, mask: Tensor) -> float:
        preds_thresholded = (preds > .5).to(torch.int)
        correct_preds = (preds_thresholded == labels).to(torch.float)
        accuracy = Masked.mean(correct_preds.mean(-1), mask).item()

        return accuracy

    @staticmethod
    def logloss(preds: Tensor, labels: Tensor, mask: Tensor) -> Tensor:
        logloss: Tensor = - labels * torch.log(preds) - (1 - labels) * torch.log(1 - preds)
        return Masked.mean(logloss.mean(-1), mask)

    @staticmethod
    def accuracy(preds: Tensor, labels: Tensor) -> float:
        preds_thresholded = (preds > .5).to(torch.int)
        correct_preds = (preds_thresholded == labels).to(torch.float)
        accuracy = correct_preds.mean().item()

        return accuracy


def concat_subproc_batch(batches: DataBatch, exclude: List[str] = None) -> AgentDataBatch:
    """Concatenate multiple sets of data in a single batch"""
    if exclude is None:
        exclude = ["__all__"]

    batches = transpose_batch(batches)

    batches = {key: value for key, value in batches.items() if key not in exclude}
    agents = list(batches.keys())

    merged = {}
    for key in batches[agents[0]]:
        merged[key] = torch.cat([batch[key] for batch in batches.values()], dim=1)

    return merged


def get_episode_lens(done_batch: Tensor) -> Tuple[int]:
    """
    Based on the recorded done values, returns the length of each episode in a batch.
    Args:
        done_batch: boolean tensor which values indicate terminal episodes

    Returns:
        tuple of episode lengths
    """
    episode_indices = done_batch.cpu().cumsum(dim=0)[:-1]
    episode_indices = torch.cat([torch.tensor([0]), episode_indices])  # [0, 0, 0, ..., 1, 1, ..., 2, ..., ...]

    ep_ids, ep_lens_tensor = torch.unique(episode_indices, return_counts=True)
    ep_lens = tuple(ep_lens_tensor.cpu().numpy())

    return ep_lens


def get_episode_rewards(batch: DataBatch) -> np.ndarray:
    """Computes the total reward in each episode in a data batch"""
    batch = transpose_batch(batch)['Agent0']
    ep_lens = get_episode_lens(batch['dones'])

    ep_rewards = np.array([torch.sum(rewards) for rewards in torch.split(batch['rewards'], ep_lens)])

    return ep_rewards


def batch_to_gpu(data_batch: AgentDataBatch) -> AgentDataBatch:
    new_batch = {}
    for key in data_batch:
        if key == 'states':
            new_batch[key] = tuple(state_.cuda() for state_ in data_batch[key])
        else:
            new_batch[key] = data_batch[key].cuda()
    return new_batch


def minibatches(data: Dict[str, Tensor], batch_size: int, shuffle: bool = True) -> Tuple[Tensor, Dict[str, Tensor]]:
    batch_start = 0
    batch_end = batch_size
    data_size = len(data['dones'])

    if shuffle:
        indices = torch.randperm(data_size)
        data = {k: val[indices] for k, val in data.items()}
    else:
        indices = torch.arange(data_size)

    while batch_start < data_size:
        batch = {key: value[batch_start:batch_end] for key, value in data.items()}

        batch_start = batch_end
        batch_end = min(batch_start + batch_size, data_size)

        yield indices[batch_start:batch_end], batch


def pack(dict_: Dict[str, Observation]) -> Tuple[Observation, List[str]]:
    keys = list(dict_.keys())
    values = Observation.stack_tensor([dict_[key] for key in keys])

    return values, keys


def unpack(arrays: Any, keys: List[str]) -> Dict[str, Any]:
    value_dict = {key: arrays[i] for i, key in enumerate(keys)}
    return value_dict


def parse_agent_name(name: str) -> Dict[str, str]:
    parts = name.split('&')
    result = {
        "name": parts[0]
    }
    for part in parts[1:]:
        subname, value = part.split('=')
        result[subname] = value

    return result


def split_ana(content: str) -> List[str]:
    """
    Splits an .ana file into parts, each of which is a string that corresponds to some data batch.
    """
    outputs = []
    temp = []
    for line in content.split('\n'):
        if len(line) > 0 and line[0] != ' ':
            outputs.append('\n'.join(temp))
            temp = [line]
        else:
            temp.append(line)

    outputs.append('\n'.join(temp))
    return outputs[1:]


def parse_segment(content: str) -> Tuple[str, np.ndarray]:
    """Parses a segment of .ana data.
    The first line is assumed to be the title, each line after that has one or more numbers"""
    result = []
    lines = content.split('\n')
    name = lines[0]
    for line in lines[1:-1]:
        line = line.strip()
        numbers = [float(num) for num in line.split(' ') if len(num) > 0]

        result.append(numbers)

    result = np.array(result)
    return name, result


def parse_ana(content: str) -> Dict:
    """Parse the text of the entire file, split it into segments and return a dictionary of arrays"""
    segments = split_ana(content)
    data = [parse_segment(segment) for segment in segments]
    data = {name.strip(): array for name, array in data}
    return data


def read_ana(path: str) -> Dict:
    """Same as read_ana, but handle reading the file as well"""
    with open(path, "r") as f:
        text = f.read()

    data = parse_ana(text)
    return data
