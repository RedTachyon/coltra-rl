from typing import Generic, TypeVar, Any, Callable

from gym import Space

StateType = TypeVar("StateType")
ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


class FuncMAEnv(Generic[StateType, ObsType, ActType]):
    observation_space: Space
    action_space: Space

    def initial(self, rng: Any = None) -> StateType:
        raise NotImplementedError

    def transition(self, state: StateType, action: ActType, rng: Any = None) -> StateType:
        raise NotImplementedError

    def reward(self, state: StateType, action: ActType, next_state: StateType) -> float:
        raise NotImplementedError

    def observation(self, state: StateType) -> ObsType:
        raise NotImplementedError

    def transform(self, func: Callable[[Callable], Callable]):
        self.initial = func(self.initial)
        self.transition = func(self.transition)
        self.reward = func(self.reward)
        self.observation = func(self.observation)


