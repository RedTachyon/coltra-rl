import numpy as np
import gym
from coltra.envs import MultiGymEnv
from coltra.buffers import Observation, Action

# def test_wrapper():
#     env = MultiAgentWrapper(gym.make("MountainCarContinuous-v0"))
#     obs = env.reset()
#     assert isinstance(obs, dict)
#     assert isinstance(obs['agent'], Observation)
#     assert isinstance(obs['agent'].vector, np.ndarray)
#
#     action = {env.name: Action(continuous=env.action_space.sample().astype(np.float32))}
#     obs, reward, done, info = env.step(action)
#     assert isinstance(obs, dict)
#     assert isinstance(obs['agent'], Observation)
#     assert isinstance(obs['agent'].vector, np.ndarray)
#
#     assert isinstance(reward, dict)
#     assert isinstance(reward[env.name], float)
#
#     assert isinstance(done, dict)
#     assert isinstance(done[env.name], bool)


def test_multigym():
    env = MultiGymEnv.get_venv(1, "MountainCar-v0")
    obs = env.reset()

    name = list(obs.keys())[0]
    assert isinstance(obs, dict)
    assert isinstance(obs[name], Observation)
    assert isinstance(obs[name].vector, np.ndarray)

    action = {key: Action(discrete=env.action_space.sample())
              for key in obs}

    obs, reward, done, info = env.step(action)
    assert isinstance(obs, dict)
    assert isinstance(obs[name], Observation)
    assert isinstance(obs[name].vector, np.ndarray)

    assert isinstance(reward, dict)
    assert isinstance(reward[name], float)

    assert isinstance(done, dict)
    assert isinstance(done[name], bool)
