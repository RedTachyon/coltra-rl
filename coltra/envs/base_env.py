from abc import ABC, abstractmethod
import inspect
import pickle
from typing import Sequence, Optional, List, Union, Tuple, Dict, Any, Type, Iterable

import cloudpickle
import gym
import numpy as np

from coltra.buffers import Observation, Action, Reward, Done


ObsDict = Dict[str, Observation]
ActionDict = Dict[str, Action]
RewardDict = Dict[str, Reward]
DoneDict = Dict[str, Done]
InfoDict = Dict[str, Any]

StepReturn = Tuple[ObsDict, RewardDict, DoneDict, InfoDict]
VecEnvIndices = Union[None, int, Iterable[int]]

class AlreadySteppingError(Exception):
    """
    Raised when an asynchronous step is running while
    step_async() is called again.
    """

    def __init__(self):
        msg = 'already running an async step'
        Exception.__init__(self, msg)


class NotSteppingError(Exception):
    """
    Raised when an asynchronous step is not running but
    step_wait() is called.
    """

    def __init__(self):
        msg = 'not running an async step'
        Exception.__init__(self, msg)


class VecEnv(ABC):
    """
    An abstract asynchronous, vectorized environment.

    :param num_envs: (int) the number of environments
    :param observation_space: (Gym Space) the observation space
    :param action_space: (Gym Space) the action space
    """
    metadata = {
        'render.modes': ['human', 'rgb_array']
    }

    def __init__(self, num_envs, observation_space, action_space):
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.action_space = action_space

    @abstractmethod
    def reset(self):
        """
        Reset all the environments and return an array of
        observations, or a tuple of observation arrays.

        If step_async is still doing work, that work will
        be cancelled and step_wait() should not be called
        until step_async() is invoked again.

        :return: ([int] or [float]) observation
        """
        pass

    @abstractmethod
    def step_async(self, actions):
        """
        Tell all the environments to start taking a step
        with the given actions.
        Call step_wait() to get the results of the step.

        You should not call this if a step_async run is
        already pending.
        """
        pass

    @abstractmethod
    def step_wait(self):
        """
        Wait for the step taken with step_async().

        :return: ([int] or [float], [float], [bool], dict) observation, reward, done, information
        """
        pass

    @abstractmethod
    def close(self):
        """
        Clean up the environment's resources.
        """
        pass

    @abstractmethod
    def get_attr(self, attr_name, indices=None):
        """
        Return attribute from vectorized environment.

        :param attr_name: (str) The name of the attribute whose value to return
        :param indices: (list,int) Indices of envs to get attribute from
        :return: (list) List of values of 'attr_name' in all environments
        """
        pass

    @abstractmethod
    def set_attr(self, attr_name, value, indices=None):
        """
        Set attribute inside vectorized environments.

        :param attr_name: (str) The name of attribute to assign new value
        :param value: (obj) Value to assign to `attr_name`
        :param indices: (list,int) Indices of envs to assign value
        :return: (NoneType)
        """
        pass

    @abstractmethod
    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        """
        Call instance methods of vectorized environments.

        :param method_name: (str) The name of the environment method to invoke.
        :param indices: (list,int) Indices of envs whose method to call
        :param method_args: (tuple) Any positional arguments to provide in the call
        :param method_kwargs: (dict) Any keyword arguments to provide in the call
        :return: (list) List of items returned by the environment's method call
        """
        pass

    @abstractmethod
    def seed(self, seed: Optional[int] = None) -> List[Union[None, int]]:
        """
        Sets the random seeds for all environments, based on a given seed.
        Each individual environment will still get its own seed, by incrementing the given seed.

        :param seed: (Optional[int]) The random seed. May be None for completely random seeding.
        :return: (List[Union[None, int]]) Returns a list containing the seeds for each individual env.
            Note that all list elements may be None, if the env does not return anything when being seeded.
        """
        pass

    def step(self, actions):
        """
        Step the environments with the given action

        :param actions: ([int] or [float]) the action
        :return: ([int] or [float], [float], [bool], dict) observation, reward, done, information
        """
        self.step_async(actions)
        return self.step_wait()

    def get_images(self) -> Sequence[np.ndarray]:
        """
        Return RGB images from each environment
        """
        raise NotImplementedError

    def render(self, mode: str = 'human'):
        return

    def getattr_depth_check(self, name, already_found):
        """Check if an attribute reference is being hidden in a recursive call to __getattr__

        :param name: (str) name of attribute to check for
        :param already_found: (bool) whether this attribute has already been found in a wrapper
        :return: (str or None) name of module whose attribute is being shadowed, if any.
        """
        if hasattr(self, name) and already_found:
            return "{0}.{1}".format(type(self).__module__, type(self).__name__)
        else:
            return None

    def _get_indices(self, indices):
        """
        Convert a flexibly-typed reference to environment indices to an implied list of indices.

        :param indices: (None,int,Iterable) refers to indices of envs.
        :return: (list) the implied list of indices.
        """
        if indices is None:
            indices = range(self.num_envs)
        elif isinstance(indices, int):
            indices = [indices]
        return indices


class CloudpickleWrapper(object):
    def __init__(self, var):
        """
        Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)

        :param var: (Any) the variable you wish to wrap for pickling with cloudpickle
        """
        self.var = var

    def __getstate__(self):
        return cloudpickle.dumps(self.var)

    def __setstate__(self, obs):
        self.var = cloudpickle.loads(obs)


class MultiAgentEnv(gym.Env):
    """
    Base class for a gym-like environment for multiple agents. An agent is identified with its id (string),
    and most interactions are communicated through that API (actions, states, etc)
    """

    obs_vector_size: int
    action_vector_size: int

    def __init__(self, *args, **kwargs):
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

    def render(self, mode='human'):
        raise NotImplementedError

    @staticmethod
    def pack(dict_: Dict[str, Observation]) -> Tuple[Observation, List[str]]:
        keys = list(dict_.keys())
        values = Observation.stack_tensor([dict_[key] for key in keys])

        return values, keys

    @staticmethod
    def unpack(arrays: Any, keys: List[str]) -> Dict[str, Any]:
        value_dict = {key: arrays[i] for i, key in enumerate(keys)}
        return value_dict

    @classmethod
    def get_env_creator(cls, *args, **kwargs):
        def _inner():
            return cls(*args, **kwargs)
        return _inner

    @classmethod
    def get_venv(cls, workers: int = 8, *args, **kwargs) -> VecEnv:
        raise NotImplementedError


class VecEnvWrapper(VecEnv):
    """
    Vectorized environment base class
    :param venv: the vectorized environment to wrap
    :param observation_space: the observation space (can be None to load from venv)
    :param action_space: the action space (can be None to load from venv)
    """

    def __init__(
        self,
        venv: VecEnv,
        observation_space: Optional[gym.spaces.Space] = None,
        action_space: Optional[gym.spaces.Space] = None,
    ):
        self.venv = venv
        VecEnv.__init__(
            self,
            num_envs=venv.num_envs,
            observation_space=observation_space or venv.observation_space,
            action_space=action_space or venv.action_space,
        )
        self.class_attributes = dict(inspect.getmembers(self.__class__))

    def step_async(self, actions: np.ndarray) -> None:
        self.venv.step_async(actions)

    @abstractmethod
    def reset(self) -> ObsDict:
        pass

    @abstractmethod
    def step_wait(self) -> StepReturn:
        pass

    def seed(self, seed: Optional[int] = None) -> List[Union[None, int]]:
        return self.venv.seed(seed)

    def close(self) -> None:
        return self.venv.close()

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        return self.venv.render(mode=mode)

    def get_images(self) -> Sequence[np.ndarray]:
        return self.venv.get_images()

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]:
        return self.venv.get_attr(attr_name, indices)

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        return self.venv.set_attr(attr_name, value, indices)

    def env_method(self, method_name: str, *method_args, indices: VecEnvIndices = None, **method_kwargs) -> List[Any]:
        return self.venv.env_method(method_name, *method_args, indices=indices, **method_kwargs)

    def env_is_wrapped(self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices = None) -> List[bool]:
        return self.venv.env_is_wrapped(wrapper_class, indices=indices)

    def __getattr__(self, name: str) -> Any:
        """Find attribute from wrapped venv(s) if this wrapper does not have it.
        Useful for accessing attributes from venvs which are wrapped with multiple wrappers
        which have unique attributes of interest.
        """
        blocked_class = self.getattr_depth_check(name, already_found=False)
        if blocked_class is not None:
            own_class = f"{type(self).__module__}.{type(self).__name__}"
            error_str = (
                f"Error: Recursive attribute lookup for {name} from {own_class} is "
                "ambiguous and hides attribute from {blocked_class}"
            )
            raise AttributeError(error_str)

        return self.getattr_recursive(name)

    def _get_all_attributes(self) -> Dict[str, Any]:
        """Get all (inherited) instance and class attributes
        :return: all_attributes
        """
        all_attributes = self.__dict__.copy()
        all_attributes.update(self.class_attributes)
        return all_attributes

    def getattr_recursive(self, name: str) -> Any:
        """Recursively check wrappers to find attribute.
        :param name: name of attribute to look for
        :return: attribute
        """
        all_attributes = self._get_all_attributes()
        if name in all_attributes:  # attribute is present in this wrapper
            attr = getattr(self, name)
        elif hasattr(self.venv, "getattr_recursive"):
            # Attribute not present, child is wrapper. Call getattr_recursive rather than getattr
            # to avoid a duplicate call to getattr_depth_check.
            attr = self.venv.getattr_recursive(name)
        else:  # attribute not present, child is an unwrapped VecEnv
            attr = getattr(self.venv, name)

        return attr

    def getattr_depth_check(self, name: str, already_found: bool) -> str:
        """See base class.
        :return: name of module whose attribute is being shadowed, if any.
        """
        all_attributes = self._get_all_attributes()
        if name in all_attributes and already_found:
            # this venv's attribute is being hidden because of a higher venv.
            shadowed_wrapper_class = f"{type(self).__module__}.{type(self).__name__}"
        elif name in all_attributes and not already_found:
            # we have found the first reference to the attribute. Now check for duplicates.
            shadowed_wrapper_class = self.venv.getattr_depth_check(name, True)
        else:
            # this wrapper does not have the attribute. Keep searching.
            shadowed_wrapper_class = self.venv.getattr_depth_check(name, already_found)

        return shadowed_wrapper_class