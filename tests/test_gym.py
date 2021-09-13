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
from coltra.models import FancyMLPModel
from coltra.trainers import PPOCrowdTrainer


def test_multigym():
    env = MultiGymEnv.get_venv(1, "MountainCar-v0")
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
  input_size:
  num_actions:
  activation: tanh
  discrete:

  hidden_sizes: [16, 16]
  separate_value: true

  rays_input_size:

  conv_filters:

  sigma0: 0.7

  vec_input_size: 4
  rel_input_size: 4
  vec_hidden_layers: [32, 32]
  rel_hidden_layers: [32, 32]
  com_hidden_layers: [32, 32]

  initializer: xavier_uniform"""

    config = yaml.load(CONFIG, yaml.Loader)
    trainer_config = config["trainer"]
    model_config = config["model"]
    workers = 2

    env = MultiGymEnv.get_venv(workers, "MountainCar-v0")
    action_space = env.action_space

    print(f"{env.observation_space=}")
    print(f"{action_space=}")

    is_discrete_action = isinstance(action_space, gym.spaces.Discrete)
    if is_discrete_action:
        action_shape = action_space.n
    else:
        action_shape = action_space.shape[0]

    # Initialize the agent
    sample_obs = next(iter(env.reset().values()))
    obs_size = sample_obs.vector.shape[0]
    ray_size = sample_obs.rays.shape[0] if sample_obs.rays is not None else None

    model_config["input_size"] = obs_size
    model_config["rays_input_size"] = ray_size
    model_config["discrete"] = is_discrete_action
    model_config["num_actions"] = action_shape

    model_cls = FancyMLPModel
    agent_cls = CAgent if isinstance(action_space, gym.spaces.Box) else DAgent

    model = model_cls(model_config)
    agent = agent_cls(model)

    trainer = PPOCrowdTrainer(agent, env, config)
    trainer.train(2, disable_tqdm=False, save_path=None)
