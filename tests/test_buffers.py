import pytest
import numpy as np
import torch

from coltra.buffers import Observation, Action, TensorArray, Reward, LogProb, Value, Done, MemoryRecord, MemoryBuffer, \
    AgentMemoryBuffer


def test_observation_array():
    obs = Observation(vector=np.random.randn(5, 81), buffer=np.random.randn(5, 10, 4))
    assert obs.batch_size == 5
    assert obs.vector.shape == (5, 81)
    assert obs.buffer.shape == (5, 10, 4)


def test_observation_tensor():
    obs = Observation(vector=torch.randn(5, 81), buffer=torch.randn(5, 10, 4))
    assert obs.batch_size == 5
    assert obs.vector.shape == (5, 81)
    assert obs.buffer.shape == (5, 10, 4)


def test_observation_misshaped():
    obs = Observation(vector=torch.randn(5, 81), buffer=torch.randn(7, 10, 4))
    with pytest.raises(Exception):
        b = obs.batch_size


def test_obs_to_tensor():
    obs = Observation(vector=np.random.randn(5, 81), buffer=np.random.randn(5, 10, 4)).tensor()
    assert obs.batch_size == 5
    assert obs.vector.shape == (5, 81)
    assert obs.buffer.shape == (5, 10, 4)


def test_obs_get():
    obs = Observation(vector=np.random.randn(5, 81), buffer=np.random.randn(5, 10, 4))

    part_obs = obs[0:2]  # [0, 1]

    assert part_obs.batch_size == 2
    assert part_obs.vector.shape == (2, 81)
    assert part_obs.buffer.shape == (2, 10, 4)


def test_obs_stack():
    obs_list = [Observation(vector=np.random.randn(81), buffer=np.random.randn(10, 4))
                for _ in range(7)]

    obs = Observation.stack_tensor(obs_list, dim=0)

    assert obs.batch_size == 7
    assert obs.vector.shape == (7, 81)
    assert obs.buffer.shape == (7, 10, 4)


def test_obs_cat():
    obs_list = [Observation(vector=np.random.randn(5, 81), buffer=np.random.randn(5, 10, 4))
                for _ in range(5)]

    obs = Observation.cat_tensor(obs_list, dim=0)

    assert obs.batch_size == 25
    assert obs.vector.shape == (25, 81)
    assert obs.buffer.shape == (25, 10, 4)


def test_action_array():
    obs = Action(continuous=np.random.randn(5, 81), discrete=np.random.randn(5, 4))
    assert obs.batch_size == 5
    assert obs.continuous.shape == (5, 81)
    assert obs.discrete.shape == (5, 4)


def test_action_tensor():
    obs = Action(continuous=torch.randn(5, 81), discrete=torch.randn(5, 4))
    assert obs.batch_size == 5
    assert obs.continuous.shape == (5, 81)
    assert obs.discrete.shape == (5, 4)


def test_action_misshaped():
    obs = Action(continuous=torch.randn(5, 81), discrete=torch.randn(7, 4))
    with pytest.raises(Exception):
        b = obs.batch_size


def test_apply():
    obs = Observation(vector=np.ones((5, 81)), buffer=np.ones((5, 10, 4)))
    new_obs = obs.apply(lambda x: 2*x)

    assert np.allclose(new_obs.vector, 2*np.ones((5, 81)))
    assert np.allclose(new_obs.buffer, 2*np.ones((5, 10, 4)))


def test_memory_buffer():
    memory = MemoryBuffer()
    agents = ["Agent1", "Agent2", "Agent3"]
    batch_size = 100

    for _ in range(batch_size):
        obs = {agent_id: Observation(vector=np.random.randn(81).astype(np.float32),
                                     buffer=np.random.randn(10, 4).astype(np.float32)) for agent_id in agents}
        action = {agent_id: Action(continuous=np.random.randn(2).astype(np.float32)) for agent_id in agents}
        reward = {agent_id: np.random.randn(1).astype(np.float32) for agent_id in agents}
        value = {agent_id: np.random.randn(1).astype(np.float32) for agent_id in agents}
        done = {agent_id: False for agent_id in agents}

        memory.append(obs, action, reward, value, done)

    data = memory.tensorify()
    assert isinstance(data, dict)
    assert isinstance(data["Agent1"], MemoryRecord)
    assert data["Agent1"].obs.vector.shape == (batch_size, 81)
    assert data["Agent1"].obs.buffer.shape == (batch_size, 10, 4)
    assert data["Agent1"].obs.batch_size == batch_size

    crowd_data = memory.crowd_tensorify()
    assert isinstance(crowd_data, MemoryRecord)
    assert crowd_data.obs.vector.shape == (3*batch_size, 81)
    assert crowd_data.obs.buffer.shape == (3*batch_size, 10, 4)
    assert crowd_data.obs.batch_size == 3*batch_size


