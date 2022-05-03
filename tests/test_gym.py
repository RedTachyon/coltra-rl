import numpy as np
import gym
import yaml

from coltra.agents import CAgent, DAgent
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
from coltra.groups import HomogeneousGroup
from coltra.models import MLPModel
from coltra.trainers import PPOCrowdTrainer


def test_multigym_single():
    env = MultiGymEnv(env_name="MountainCar-v0")
    obs = env.reset()

    name = list(obs.keys())[0]
    assert isinstance(obs, dict)
    assert isinstance(obs[name], Observation)
    assert isinstance(obs[name].vector, np.ndarray)

    action = {key: Action(discrete=env.action_space.sample()) for key in obs}

    obs, reward, done, info = env.step(action)
    assert isinstance(obs, dict)
    assert isinstance(obs[name], Observation)
    assert isinstance(obs[name].vector, np.ndarray)

    assert isinstance(reward, dict)
    assert isinstance(reward[name], float)

    assert isinstance(done, dict)
    assert isinstance(done[name], bool)


def test_multigym():
    env = MultiGymEnv.get_venv(workers=1, env_name="MountainCar-v0")
    obs = env.reset()

    # name = list(obs.keys())[0]
    # assert isinstance(obs, dict)
    # assert isinstance(obs[name], Observation)
    # assert isinstance(obs[name].vector, np.ndarray)
    #
    # action = {key: Action(discrete=env.action_space.sample()) for key in obs}
    #
    # obs, reward, done, info = env.step(action)
    # assert isinstance(obs, dict)
    # assert isinstance(obs[name], Observation)
    # assert isinstance(obs[name].vector, np.ndarray)
    #
    # assert isinstance(reward, dict)
    # assert isinstance(reward[name], float)
    #
    # assert isinstance(done, dict)
    # assert isinstance(done[name], bool)


def test_training():

    CONFIG = """---
trainer:
  steps: 2
  workers: 2

  tensorboard_name:
  save_freq: 10

  PPOConfig:
    optimizer: adam
    OptimizerKwargs:
      lr: 0.0003
      betas: !!python/tuple [0.9, 0.999]
      eps: 1.0e-07
      weight_decay: 0
      amsgrad: false

    gamma: 0.99
    eta: 0.0
    gae_lambda: 0.95

    advantage_normalization: true

    eps: 0.25
    target_kl: 0.03
    entropy_coeff: 0.0
    entropy_decay_time: 100
    min_entropy: 0.0
    value_coeff: 1.0

    ppo_epochs: 10
    minibatch_size: 8000

    use_gpu: false

model:
  activation: tanh

  hidden_sizes: [16, 16]
  separate_value: true

  conv_filters:

  sigma0: 0.7

  vec_hidden_layers: [32, 32]
  rel_hidden_layers: [32, 32]
  com_hidden_layers: [32, 32]

  initializer: xavier_uniform"""

    config = yaml.load(CONFIG, yaml.Loader)
    trainer_config = config["trainer"]
    model_config = config["model"]
    workers = 2

    env = MultiGymEnv.get_venv(workers=workers, env_name="MountainCar-v0")

    print(f"{env.observation_space=}")
    print(f"{env.action_space=}")

    # Initialize the agent

    model_cls = MLPModel
    agent_cls = CAgent if isinstance(env.action_space, gym.spaces.Box) else DAgent

    model = model_cls(
        model_config,
        observation_space=env.observation_space,
        action_space=env.action_space,
    )
    agent = agent_cls(model)

    group = HomogeneousGroup(agent)

    trainer = PPOCrowdTrainer(group, env, trainer_config)
    trainer.train(2, disable_tqdm=False, save_path=None)
