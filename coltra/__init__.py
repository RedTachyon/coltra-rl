from coltra.agents import Agent, CAgent, DAgent
from coltra.groups import HomogeneousGroup
from coltra.collectors import collect_crowd_data, collect_renders
from coltra.trainers import PPOCrowdTrainer
from coltra.envs import MultiAgentEnv
from coltra.buffers import Action, Observation

__version__ = "0.1.1"
VERSION = __version__
