from typing import Tuple, Dict

from torch import Tensor

from coltra.agents import Agent
from coltra.buffers import Observation, Action
from coltra.envs import MultiAgentEnv


class AgentWrapper(Agent):
    def __init__(self, agent: Agent):
        super().__init__()
        self.agent = agent

    def act(
        self,
        obs_batch: Observation,
        state_batch: Tuple = (),
        deterministic: bool = False,
        get_value: bool = False,
        **kwargs
    ) -> Tuple[Action, Tuple, Dict]:
        return self.agent.act(obs_batch, state_batch, deterministic, get_value)

    def evaluate(
        self, obs_batch: Observation, action_batch: Action
    ) -> Tuple[Tensor, Tensor, Tensor]:
        return self.agent.evaluate(obs_batch, action_batch)

    def value(self, obs_batch: Observation, **kwargs) -> Tensor:
        return self.value(
            obs_batch,
        )

    @property
    def unwrapped(self):
        return self.agent.unwrapped

    def __getattr__(self, name: str):
        return getattr(self.agent, name)

    def __setstate__(self, state: Dict):
        # setattr(self, "agent", state["agent"])
        for key in state:
            setattr(self, key, state[key])

    def __getstate__(self) -> Dict:
        return self.__dict__


class EnvWrapper(MultiAgentEnv):
    def __init__(self, env, **kwargs):
        super().__init__(**kwargs)
        self.env = env

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(
                "attempted to get missing private attribute '{}'".format(name)
            )
        return getattr(self.env, name)

    def step(self, action):
        return self.env.step(action)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def render(self, mode="human", **kwargs):
        return self.env.render()

    @classmethod
    def get_venv(cls, workers: int = 8, *args, **kwargs) -> MultiAgentEnv:
        raise NotImplementedError
