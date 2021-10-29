import itertools

from PIL.Image import Image
import numpy as np
from typing import Any, Dict, List, Tuple, Union, Optional
import time
import gym
from gym import error, spaces

from mlagents_envs.base_env import ActionTuple, BaseEnv
from mlagents_envs.base_env import DecisionSteps, TerminalSteps
from mlagents_envs import logging_util
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import (
    EngineConfigurationChannel,
)
from mlagents_envs.side_channel.environment_parameters_channel import (
    EnvironmentParametersChannel,
)

from coltra.buffers import Action, Observation
from coltra.envs import MultiAgentEnv, SubprocVecEnv
from coltra.envs.base_env import ActionDict, VecEnv
from coltra.utils import np_float


class SmartNavEnv(MultiAgentEnv):

    def __init__(
        self,
        file_name: Optional[str] = None,
        seed: Optional[int] = None,
        metrics: Optional[list[str]] = None,
        env_params: Optional[dict[str, Any]] = None,
        time_scale: float = 100.0,
        virtual_display: Optional[tuple[int, int]] = None,
        **kwargs,
    ):
        super().__init__(seed, **kwargs)
        if env_params is None:
            env_params = {}
        if metrics is None:
            metrics = []

        if virtual_display:
            from pyvirtualdisplay.smartdisplay import SmartDisplay

            self.virtual_display = SmartDisplay(size=virtual_display)
            self.virtual_display.start()
        else:
            self.virtual_display = None

        self.metrics = metrics
        self.num_metrics = len(self.metrics)
        self.path = file_name

        self.engine_channel = EngineConfigurationChannel()
        self.param_channel = EnvironmentParametersChannel()

        channels = [self.engine_channel, self.param_channel]

        self.unity = UnityEnvironment(self.path, side_channels=channels, **kwargs)
        self.env = UnityToGymWrapper(self.unity)

        self.engine_channel.set_configuration_parameters(time_scale=time_scale)

        for key in env_params:
            self.param_channel.set_float_parameter(key, env_params[key])

        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def reset(self, **kwargs):
        for key in kwargs:
            self.param_channel.set_float_parameter(key, kwargs[key])
        obs = self.env.reset()
        obs, _ = self.process_obs(obs)
        return obs

    def step(self, action_dict: ActionDict):
        obs, reward, done, _ = self.env.step(self.process_action(action_dict))
        if all(done.values()):
            done["__all__"] = True
        else:
            done["__all__"] = False

        obs, info = self.process_obs(obs)

        return obs, reward, done, info

    def process_action(self, action_dict: dict[str, Action]):
        return {agent_id: action_dict[agent_id].continuous for agent_id in action_dict}

    def process_obs(
        self, obs_dict: dict[str, np.ndarray]
    ) -> Tuple[dict[str, Observation], dict]:
        all_obs = {}
        all_info = {}
        for agent_id in obs_dict:
            obs, info = self.process_single_obs(obs_dict[agent_id])
            all_obs[agent_id] = obs
            for key in info:
                all_info.setdefault(key, []).append(info[key])

        all_info = {
            f"m_{key}": np.mean(all_info[key], keepdims=True) for key in all_info
        }

        return all_obs, all_info

    def process_single_obs(self, obs: np.ndarray) -> Tuple[Observation, dict]:
        if self.num_metrics == 0:
            return Observation(obs), {}
        n_obs = Observation(vector=obs[: -self.num_metrics])
        info = {
            self.metrics[i]: np_float(obs[-self.num_metrics + i])
            for i in range(self.num_metrics)
        }
        return n_obs, info

    def render(self, mode="rgb_array") -> Optional[Union[np.ndarray, Image]]:
        if self.virtual_display:
            img = self.virtual_display.grab()
            if mode == "rgb_array":
                return np.array(img)
            else:
                return img
        else:
            return None

    @classmethod
    def get_venv(
        cls, workers: int = 8, file_name: Optional[str] = None, *args, **kwargs
    ) -> SubprocVecEnv:
        venv = SubprocVecEnv(
            [
                cls.get_env_creator(
                    file_name=file_name,
                    worker_id=i,
                    seed=i,
                    **kwargs,
                )
                for i in range(workers)
            ]
        )
        return venv

    def close(self):
        self.env.close()


class UnityGymException(error.Error):
    """
    Any error related to the gym wrapper of ml-agents.
    """

    pass


logger = logging_util.get_logger(__name__)
logging_util.set_log_level(logging_util.INFO)

GymStepResult = Tuple[np.ndarray, float, bool, Dict]


class UnityToGymWrapper(gym.Env):
    """
    Provides Gym wrapper for Unity Learning Environments. From RLLib.
    """

    def __init__(
        self,
        unity_env: BaseEnv,
        uint8_visual: bool = False,
        flatten_branched: bool = False,
        allow_multiple_obs: bool = False,
    ):
        """
        Environment initialization
        :param unity_env: The Unity BaseEnv to be wrapped in the gym. Will be closed when the UnityToGymWrapper closes.
        :param uint8_visual: Return visual observations as uint8 (0-255) matrices instead of float (0.0-1.0).
        :param flatten_branched: If True, turn branched discrete action spaces into a Discrete space rather than
            MultiDiscrete.
        :param allow_multiple_obs: If True, return a list of np.ndarrays as observations with the first elements
            containing the visual observations and the last element containing the array of vector observations.
            If False, returns a single np.ndarray containing either only a single visual observation or the array of
            vector observations.
        """
        self._env = unity_env

        # Take a single step so that the brain information will be sent over
        if not self._env.behavior_specs:
            self._env.step()

        self.visual_obs = None

        # Save the step result from the last time all Agents requested decisions.
        self._previous_decision_step: Optional[DecisionSteps] = None
        self._flattener = None
        # Hidden flag used by Atari environments to determine if the game is over
        self.game_over = False
        self._allow_multiple_obs = allow_multiple_obs

        # Check brain configuration
        if len(self._env.behavior_specs) != 1:
            raise UnityGymException(
                "There can only be one behavior in a UnityEnvironment "
                "if it is wrapped in a gym."
            )

        self.name = list(self._env.behavior_specs.keys())[0]
        self.group_spec = self._env.behavior_specs[self.name]

        if self._get_n_vis_obs() == 0 and self._get_vec_obs_size() == 0:
            raise UnityGymException(
                "There are no observations provided by the environment."
            )

        if not self._get_n_vis_obs() >= 1 and uint8_visual:
            logger.warning(
                "uint8_visual was set to true, but visual observations are not in use. "
                "This setting will not have any effect."
            )
        else:
            self.uint8_visual = uint8_visual
        if (
            self._get_n_vis_obs() + self._get_vec_obs_size() >= 2
            and not self._allow_multiple_obs
        ):
            logger.warning(
                "The environment contains multiple observations. "
                "You must define allow_multiple_obs=True to receive them all. "
                "Otherwise, only the first visual observation (or vector observation if"
                "there are no visual observations) will be provided in the observation."
            )

        # Check for number of agents in scene.
        self._env.reset()
        decision_steps, _ = self._env.get_steps(self.name)
        self._num_agents = len(decision_steps)
        self._check_agents(len(decision_steps))
        self._previous_decision_step = decision_steps

        # Set action spaces
        if self.group_spec.action_spec.is_discrete():
            self.action_size = self.group_spec.action_spec.discrete_size
            branches = self.group_spec.action_spec.discrete_branches
            if self.group_spec.action_spec.discrete_size == 1:
                self._action_space = spaces.Discrete(branches[0])
            else:
                if flatten_branched:
                    self._flattener = ActionFlattener(branches)
                    self._action_space = self._flattener.action_space
                else:
                    self._action_space = spaces.MultiDiscrete(branches)

        elif self.group_spec.action_spec.is_continuous():
            if flatten_branched:
                logger.warning(
                    "The environment has a non-discrete action space. It will "
                    "not be flattened."
                )

            self.action_size = self.group_spec.action_spec.continuous_size
            high = np.array([1] * self.group_spec.action_spec.continuous_size)
            self._action_space = spaces.Box(-high, high, dtype=np.float32)
        else:
            raise UnityGymException(
                "The gym wrapper does not provide explicit support for both discrete "
                "and continuous actions."
            )

        # Set observations space
        list_spaces: List[gym.Space] = []
        shapes = self._get_vis_obs_shape()
        for shape in shapes:
            if uint8_visual:
                list_spaces.append(spaces.Box(0, 255, dtype=np.uint8, shape=shape))
            else:
                list_spaces.append(spaces.Box(0, 1, dtype=np.float32, shape=shape))
        if self._get_vec_obs_size() > 0:
            # vector observation is last
            high = np.array([np.inf] * self._get_vec_obs_size())
            list_spaces.append(spaces.Box(-high, high, dtype=np.float32))
        if self._allow_multiple_obs:
            self._observation_space = spaces.Tuple(list_spaces)
        else:
            self._observation_space = list_spaces[0]  # only return the first one

    def reset(self):
        """Resets the entire Unity3D scene (a single multi-agent episode)."""
        self._env.reset()
        obs, _, _, _ = self._get_step_results()
        return obs

    def step(self, action_dict):
        """Performs one multi-agent step through the game.
        Args:
            action_dict (dict): Multi-agent action dict with:
                keys=agent identifier consisting of
                [MLagents behavior name, e.g. "Goalie?team=1"] + "_" +
                [Agent index, a unique MLAgent-assigned index per single agent]
        Returns:
            tuple:
                - obs: Multi-agent observation dict.
                    Only those observations for which to get new actions are
                    returned.
                - rewards: Rewards dict matching `obs`.
                - dones: Done dict with only an __all__ multi-agent entry in
                    it. __all__=True, if episode is done for all agents.
                - infos: An (empty) info dict.
        """

        # Set only the required actions (from the DecisionSteps) in Unity3D.
        all_agents = []
        for behavior_name in self._env.behavior_specs:
            actions = []
            for agent_id in self._env.get_steps(behavior_name)[0].agent_id:
                key = behavior_name + "_{}".format(agent_id)
                all_agents.append(key)
                actions.append(action_dict[key])
            if actions:
                actions_array = np.array(actions)
                if actions_array.ndim == 1:
                    actions_array = np.expand_dims(actions_array, 0)
                if actions_array[0].dtype == np.float32:
                    action_tuple = ActionTuple(continuous=np.array(actions_array))
                else:
                    action_tuple = ActionTuple(discrete=np.array(actions_array))
                self._env.set_actions(behavior_name, action_tuple)
        # Do the step.
        self._env.step()

        return self._get_step_results()

    def _get_step_results(self):
        """Collects those agents' obs/rewards that have to act in next `step`.

        Returns:
            Tuple:
                obs: Multi-agent observation dict.
                    Only those observations for which to get new actions are
                    returned.
                rewards: Rewards dict matching `obs`.
                dones: Done dict with only an __all__ multi-agent entry in it.
                    __all__=True, if episode is done for all agents.
                infos: An (empty) info dict.
        """
        obs = {}
        rewards = {}
        infos = {}
        done = {}
        terminal_agents = {}
        for behavior_name in self._env.behavior_specs:
            decision_steps, terminal_steps = self._env.get_steps(behavior_name)

            # Sometimes we dont get all agents information
            # take env steps until we do
            # track terminal agents and their rewards during the loop
            start_time = time.time()
            while len(decision_steps) + len(terminal_steps) < self._num_agents:
                if time.time() - start_time > 60:
                    raise ValueError(
                        "Environment does not have the same amount of agents"
                    )
                for agent_id, idx in terminal_steps.agent_id_to_index.items():
                    key = behavior_name + "_{}".format(agent_id)
                    terminal_agents[key] = terminal_steps.reward[idx]
                self._env.step()
                decision_steps, terminal_steps = self._env.get_steps(behavior_name)

            for step_info in [decision_steps, terminal_steps]:
                for agent_id, idx in step_info.agent_id_to_index.items():
                    key = behavior_name + "_{}".format(agent_id)
                    os = tuple(o[idx] for o in step_info.obs)
                    os = os[0] if len(os) == 1 else os
                    obs[key] = os
                    if type(step_info) is TerminalSteps or key not in terminal_agents:
                        rewards[key] = step_info.reward[idx]
                    else:
                        rewards[key] = terminal_agents[key]
                    done[key] = (
                        True
                        if type(step_info) is TerminalSteps or key in terminal_agents
                        else False
                    )

        return obs, rewards, done, infos

    def _preprocess_single(self, single_visual_obs: np.ndarray) -> np.ndarray:
        if self.uint8_visual:
            return (255.0 * single_visual_obs).astype(np.uint8)
        else:
            return single_visual_obs

    def _get_n_vis_obs(self) -> int:
        result = 0
        for obs_spec in self.group_spec.observation_specs:
            if len(obs_spec.shape) == 3:
                result += 1
        return result

    def _get_vis_obs_shape(self) -> List[Tuple]:
        result: List[Tuple] = []
        for obs_spec in self.group_spec.observation_specs:
            if len(obs_spec.shape) == 3:
                result.append(obs_spec.shape)
        return result

    def _get_vis_obs_list(
        self, step_result: Union[DecisionSteps, TerminalSteps]
    ) -> List[np.ndarray]:
        result: List[np.ndarray] = []
        for obs in step_result.obs:
            if len(obs.shape) == 4:
                result.append(obs)
        return result

    def _get_vector_obs(
        self, step_result: Union[DecisionSteps, TerminalSteps]
    ) -> np.ndarray:
        result: List[np.ndarray] = []
        for obs in step_result.obs:
            if len(obs.shape) == 2:
                result.append(obs)
        return np.concatenate(result, axis=1)

    def _get_vec_obs_size(self) -> int:
        result = 0
        for obs_spec in self.group_spec.observation_specs:
            if len(obs_spec.shape) == 1:
                result += obs_spec.shape[0]
        return result

    def render(self, mode="rgb_array"):
        return self.visual_obs

    def close(self) -> None:
        """Override _close in your subclass to perform any necessary cleanup.
        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        self._env.close()

    def seed(self, seed: Any = None) -> None:
        """Sets the seed for this env's random number generator(s).
        Currently not implemented.
        """
        logger.warning("Could not seed environment %s", self.name)
        return

    @staticmethod
    def _check_agents(n_agents: int) -> None:
        return
        # if n_agents > 1:
        #     raise UnityGymException(
        #         f"There can only be one Agent in the environment but {n_agents} were detected."
        #     )

    @property
    def metadata(self):
        return {"render.modes": ["rgb_array"]}

    @property
    def reward_range(self) -> Tuple[float, float]:
        return -float("inf"), float("inf")

    @property
    def spec(self):
        return None

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space


class ActionFlattener:
    """
    Flattens branched discrete action spaces into single-branch discrete action spaces.
    """

    def __init__(self, branched_action_space):
        """
        Initialize the flattener.
        :param branched_action_space: A List containing the sizes of each branch of the action
        space, e.g. [2,3,3] for three branches with size 2, 3, and 3 respectively.
        """
        self._action_shape = branched_action_space
        self.action_lookup = self._create_lookup(self._action_shape)
        self.action_space = spaces.Discrete(len(self.action_lookup))

    @classmethod
    def _create_lookup(self, branched_action_space):
        """
        Creates a Dict that maps discrete actions (scalars) to branched actions (lists).
        Each key in the Dict maps to one unique set of branched actions, and each value
        contains the List of branched actions.
        """
        possible_vals = [range(_num) for _num in branched_action_space]
        all_actions = [list(_action) for _action in itertools.product(*possible_vals)]
        # Dict should be faster than List for large action spaces
        action_lookup = {
            _scalar: _action for (_scalar, _action) in enumerate(all_actions)
        }
        return action_lookup

    def lookup_action(self, action):
        """
        Convert a scalar discrete action into a unique set of branched actions.
        :param: action: A scalar value representing one of the discrete actions.
        :return: The List containing the branched actions.
        """
        return self.action_lookup[action]
