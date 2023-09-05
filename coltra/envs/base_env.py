from typing import List, Tuple, Any

from coltra.buffers import Observation, Action, Reward, Done
from coltra.envs.spaces import ActionSpace, ObservationSpace

ObsDict = dict[str, Observation]
ActionDict = dict[str, Action]
RewardDict = dict[str, Reward]
DoneDict = dict[str, Done]
InfoDict = dict[str, Any]

StepReturn = Tuple[ObsDict, RewardDict, DoneDict, InfoDict]


class MultiAgentEnv:
    """
    Base class for a gym-like environment for multiple agents. An agent is identified with its id (string),
    and most interactions are communicated through that API (actions, states, etc)
    """

    observation_space: ObservationSpace
    action_space: ActionSpace

    def __init__(self, **kwargs):
        self.config = {}
        self.active_agents: List = []

    def reset(self, *args, **kwargs) -> ObsDict:
        """
        Resets the environment and returns the state.
        Returns:
            A dictionary holding the state visible to each agent.
        """
        raise NotImplementedError

    def step(self, action_dict: ActionDict) -> StepReturn:
        """
        Executes the chosen actions for each agent and returns information about the new state.

        Args:
            action_dict: dictionary holding each agent's action

        Returns:
            states: new state for each agent
            rewards: reward obtained by each agent
            dones: whether the environment is done for each agent
            infos: any additional information
        """
        raise NotImplementedError

    def render(self, mode="rgb_array"):
        raise NotImplementedError

    def close(self):
        pass

    @staticmethod
    def pack(dict_: dict[str, Observation]) -> Tuple[Observation, List[str]]:
        keys = list(dict_.keys())
        values = Observation.stack_tensor([dict_[key] for key in keys])

        return values, keys

    @staticmethod
    def unpack(arrays: Any, keys: List[str]) -> dict[str, Any]:
        value_dict = {key: arrays[i] for i, key in enumerate(keys)}
        return value_dict

    @classmethod
    def get_env_creator(cls, **env_kwargs):
        def _inner():
            env = cls(**env_kwargs)
            return env

        return _inner

    @classmethod
    def get_venv(cls, workers: int = 8, *args, **kwargs) -> "MultiAgentEnv":
        raise NotImplementedError
