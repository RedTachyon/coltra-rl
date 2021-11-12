import socket
from sys import platform
from typing import (
    Dict,
    List,
    Union,
    Tuple,
    Any,
    Callable,
    Optional,
    Iterator,
    Type,
    Generator,
)

import numpy as np

import torch
from mlagents_envs.exception import UnityWorkerInUseException
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import time

from torch.distributions import AffineTransform
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

# DataBatch = DataBatchT = Dict[str, Dict[str, Any]]
# AgentDataBatch = Dict[str, Union[Tensor, Tuple]]
# Array = Union[Tensor, np.ndarray]


def write_dict(
    metrics: Dict[str, Union[int, float]],
    step: int,
    writer: Optional[SummaryWriter] = None,
):
    """Writes a dictionary to a tensorboard SummaryWriter"""
    if writer is not None:
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
        "sgd": SGD,
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
        "gelu": lambda x: x * F.sigmoid(1.702 * x),
    }

    if act_name not in activations.keys():
        raise ValueError(
            f"Wrong activation: {act_name} is not a valid activation function name."
        )

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
        raise ValueError(
            f"Wrong activation: {act_name} is not a valid activation function name."
        )

    return activations[act_name]


def get_initializer(init_name: str) -> Callable[[Tensor], None]:
    """Gets an initializer by name"""
    initializers = {
        "kaiming_normal": nn.init.kaiming_normal_,
        "kaiming_uniform": nn.init.kaiming_uniform_,
        "xavier_normal": nn.init.xavier_normal_,
        "xavier_uniform": nn.init.xavier_uniform_,
        "zeros": nn.init.zeros_,
        "orthogonal": nn.init.orthogonal_,
    }

    if init_name not in initializers.keys():
        raise ValueError(
            f"Wrong initialization: {init_name} is not a valid initializer name."
        )

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


class Masked:
    @staticmethod
    def mean(input_: Tensor, mask: Tensor) -> Tensor:
        """Mean of elements not covered by the mask"""
        return torch.sum(input_ * mask) / torch.sum(mask)

    @staticmethod
    def accuracy(preds: Tensor, labels: Tensor, mask: Tensor) -> float:
        preds_thresholded = (preds > 0.5).to(torch.int)
        correct_preds = (preds_thresholded == labels).to(torch.float)
        accuracy = Masked.mean(correct_preds.mean(-1), mask).item()

        return accuracy

    @staticmethod
    def logloss(preds: Tensor, labels: Tensor, mask: Tensor) -> Tensor:
        logloss: Tensor = -labels * torch.log(preds) - (1 - labels) * torch.log(
            1 - preds
        )
        return Masked.mean(logloss.mean(-1), mask)


def get_episode_lens(done_batch: Tensor) -> Tuple[int]:
    """
    Based on the recorded done values, returns the length of each episode in a batch.
    Args:
        done_batch: boolean tensor which values indicate terminal episodes

    Returns:
        tuple of episode lengths
    """
    episode_indices = done_batch.cpu().cumsum(dim=0)[:-1]
    episode_indices = torch.cat(
        [torch.tensor([0]), episode_indices]
    )  # [0, 0, 0, ..., 1, 1, ..., 2, ..., ...]

    ep_ids, ep_lens_tensor = torch.unique(episode_indices, return_counts=True)
    ep_lens = tuple(ep_lens_tensor.cpu().numpy())

    return ep_lens


def minibatches(
    data: Dict[str, Tensor], batch_size: int, shuffle: bool = True
) -> Iterator[Tuple[Tensor, Dict[str, Tensor]]]:
    batch_start = 0
    batch_end = batch_size
    data_size = len(data["dones"])

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
    parts = name.split("&")
    result = {"name": parts[0]}
    for part in parts[1:]:
        subname, value = part.split("=")
        result[subname] = value

    return result


def split_ana(content: str) -> List[str]:
    """
    Splits an .ana file into parts, each of which is a string that corresponds to some data batch.
    """
    outputs = []
    temp = []
    for line in content.split("\n"):
        if len(line) > 0 and line[0] != " ":
            outputs.append("\n".join(temp))
            temp = [line]
        else:
            temp.append(line)

    outputs.append("\n".join(temp))
    return outputs[1:]


def parse_segment(content: str) -> Tuple[str, np.ndarray]:
    """Parses a segment of .ana data.
    The first line is assumed to be the title, each line after that has one or more numbers"""
    result = []
    lines = content.split("\n")
    name = lines[0]
    for line in lines[1:-1]:
        line = line.strip()
        numbers = [float(num) for num in line.split(" ") if len(num) > 0]

        result.append(numbers)

    result = np.array(result)
    return name, result


def parse_ana(content: str) -> Dict:
    """Parse the text of the entire file, split it into segments and return a dictionary of arrays"""
    segments = split_ana(content)
    data = [parse_segment(segment) for segment in segments]
    data_dict = {name.strip(): array for name, array in data}
    return data_dict


def read_ana(path: str) -> Dict:
    """Same as read_ana, but handle reading the file as well"""
    with open(path, "r") as f:
        text = f.read()

    data = parse_ana(text)
    return data


def is_worker_free(worker_id: int, base_port: int = 5005):
    """
    Attempts to bind to the requested communicator port, checking if it is already in use.
    Returns whether the port is free.
    """
    port = base_port + worker_id
    status = True
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    if platform == "linux" or platform == "linux2":
        # On linux, the port remains unusable for TIME_WAIT=60 seconds after closing
        # SO_REUSEADDR frees the port right after closing the environment
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        s.bind(("localhost", port))
    except OSError:
        status = False
    #         raise UnityWorkerInUseException(self.worker_id)
    finally:
        s.close()

    return status


def find_free_worker(max_value: int = 100) -> int:
    """
    Finds a free worker ID.
    """
    for worker_id in range(max_value):
        if is_worker_free(worker_id):
            return worker_id

    raise UnityWorkerInUseException("All workers are in use.")


class AffineBeta(torch.distributions.TransformedDistribution):
    def __init__(self, a: Tensor, b: Tensor, low: Tensor, high: Tensor):
        self.low = torch.as_tensor(low, dtype=torch.float32)
        self.high = torch.as_tensor(high, dtype=torch.float32)
        self.a = torch.as_tensor(a, dtype=torch.float32)
        self.b = torch.as_tensor(b, dtype=torch.float32)
        self.loc = self.low
        self.scale = self.high - self.low
        super().__init__(torch.distributions.Beta(a, b), AffineTransform(self.loc, self.scale))

    @property
    def mean(self):
        return self.base_dist.mean * self.scale + self.loc

    @property
    def variance(self):
        return self.base_dist.variance * self.scale**2

    def entropy(self):
        return self.base_dist.entropy() + self.scale.log()

    def enumerate_support(self, expand=True):
        raise NotImplementedError
