import numpy as np
from typing import Tuple, List, Union, Dict, Optional, Any
from enum import Enum

from PIL.Image import Image
from gymnasium.spaces import Box
from mlagents_envs.base_env import (
    ActionTuple,
    DecisionStep,
    TerminalStep,
    DecisionSteps,
    TerminalSteps,
    ObservationSpec,
)
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.exception import UnityEnvironmentException
from mlagents_envs.side_channel.engine_configuration_channel import (  # type: ignore
    EngineConfigurationChannel,  # type: ignore
)  # type: ignore
from mlagents_envs.side_channel.environment_parameters_channel import (  # type: ignore
    EnvironmentParametersChannel,  # type: ignore
)  # type: ignore

from coltra.envs.side_channels import StatsChannel, StringChannel, AttentionChannel
from coltra.buffers import Observation, Action
from coltra.envs.subproc_vec_env import SubprocVecEnv
from coltra.envs.base_env import (
    MultiAgentEnv,
    ObsDict,
    ActionDict,
    RewardDict,
    DoneDict,
    InfoDict,
)
from coltra.envs.spaces import ObservationSpace, ActionSpace
from coltra.unity_utils import find_free_worker, disable_unity_logs


class Sensor(Enum):
    Buffer = 0
    Ray = 1
    Vector = 2

    @staticmethod
    def from_string(name: str):
        if name.startswith("StackingSensor"):
            # Format is either:
            # StackingSensor_sizeX_VectorSensor_sizeX
            # or
            # StackingSensor_sizeX_RaySensor
            name = name.split("_")[2]

        if name.lower().startswith("buffer"):
            return Sensor.Buffer
        elif name.lower().startswith("ray"):
            return Sensor.Ray
        elif name.lower().startswith("vector"):
            return Sensor.Vector
        else:
            raise ValueError(f"{name} is not a supported sensor")

    def to_string(self) -> str:
        if self == Sensor.Buffer:
            return "buffer"
        elif self == Sensor.Ray:
            return "rays"
        elif self == Sensor.Vector:
            return "vector"
        else:
            raise ValueError(
                f"You have angered the Old God Cthulhu (and need to update the sensors)"
            )


def process_decisions(
    decisions: Union[DecisionSteps, TerminalSteps],
    name: str,
    obs_specs: List[ObservationSpec],
) -> tuple[ObsDict, RewardDict, DoneDict]:
    """
    Takes in a DecisionSteps or TerminalSteps object, and returns the relevant information (observations, rewards, dones)
    """
    dec_obs, dec_ids = decisions.obs, list(decisions.agent_id)

    obs_dict = {}
    reward_dict = {}
    done_dict = {}

    for idx in dec_ids:
        agent_name = f"{name}&id={idx}"

        obs_raw = {
            Sensor.from_string(spec.name).to_string(): obs[dec_ids.index(idx)]
            for spec, obs in zip(obs_specs, dec_obs)
        }
        # if "buffer" in obs_raw:
        #     obs_raw["buffer"] = obs_raw["buffer"][~np.all(obs_raw["buffer"] == 0, axis=1)]

        if "buffer" in obs_raw:
            obs_raw["buffer_mask"] = np.all(obs_raw["buffer"] == 0, axis=1)

        obs = Observation(**obs_raw)

        # obs = Observation(
        #     **{
        #         Sensor.from_string(spec.name).to_string(): obs[dec_ids.index(idx)]
        #         for spec, obs in zip(obs_specs, dec_obs)
        #     }
        # )

        obs_dict[agent_name] = obs
        reward_dict[agent_name] = decisions.reward[dec_ids.index(idx)]
        done_dict[agent_name] = isinstance(decisions, TerminalSteps)

    return obs_dict, reward_dict, done_dict


class UnitySimpleCrowdEnv(MultiAgentEnv):
    def __init__(
        self,
        file_name: Optional[str] = None,
        virtual_display: Optional[tuple[int, int]] = None,
        worker_id: Optional[int] = None,
        extra_params: Optional[dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__()
        self._closed = False

        disable_unity_logs()

        if extra_params is None:
            extra_params = {}

        if virtual_display:
            from pyvirtualdisplay.smartdisplay import SmartDisplay
            import sys

            if sys.platform == "darwin":
                mode = "xephyr"
            else:
                mode = "xvfb"
            self.virtual_display = SmartDisplay(size=virtual_display, backend=mode)
            self.virtual_display.start()
        else:
            self.virtual_display = None

        self.engine_channel = EngineConfigurationChannel()
        self.stats_channel = StatsChannel()
        self.param_channel = EnvironmentParametersChannel()
        self.string_channel = StringChannel()
        self.attention_channel = AttentionChannel()

        self.active_agents: List[str] = []

        kwargs.setdefault("side_channels", []).append(self.engine_channel)
        kwargs["side_channels"].append(self.stats_channel)
        kwargs["side_channels"].append(self.param_channel)
        kwargs["side_channels"].append(self.string_channel)
        kwargs["side_channels"].append(self.attention_channel)

        if worker_id is None:
            worker_id = find_free_worker(500)

        print(f"Using worker id {worker_id}")
        self.unity = UnityEnvironment(
            file_name=file_name, worker_id=worker_id, **kwargs
        )
        self.behaviors = {}
        # self.manager = ""

        self._process_parameters(extra_params)

        self.unity.reset()

        behaviors = dict(self.unity.behavior_specs)
        self.behaviors = {key: value for key, value in behaviors.items()}

        # semi-hardcoded computation of obs/action spaces, slightly different api than gym
        self.behavior_name = [
            name for name in self.behaviors.keys() if not name.startswith("Family")
        ][0]
        obs_spec, action_spec = self.behaviors[self.behavior_name]
        action_size = action_spec.continuous_size

        observations_dict = {}
        for single_obs_spec in obs_spec:
            name = Sensor.from_string(single_obs_spec.name).to_string()
            observations_dict[name] = Box(
                low=-np.inf, high=np.inf, shape=single_obs_spec.shape, dtype=np.float32
            )

        self.observation_space = ObservationSpace(observations_dict)

        self.action_space = ActionSpace(
            {
                "continuous": Box(
                    low=-np.inf, high=np.inf, shape=(action_size,), dtype=np.float32
                )
            }
        )

        self.observation_spaces = {}
        self.action_spaces = {}

        for behavior_name, (obs_spec, action_spec) in self.behaviors.items():
            action_size = action_spec.continuous_size

            observations_dict = {}
            for single_obs_spec in obs_spec:
                name = Sensor.from_string(single_obs_spec.name).to_string()
                observations_dict[name] = Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=single_obs_spec.shape,
                    dtype=np.float32,
                )

            self.observation_spaces[behavior_name] = ObservationSpace(observations_dict)

            self.action_spaces[behavior_name] = ActionSpace(
                {
                    "continuous": Box(
                        low=-np.inf, high=np.inf, shape=(action_size,), dtype=np.float32
                    )
                }
            )

    def get_observation_space(self, name: str) -> ObservationSpace:
        return self.observation_spaces[name[: name.find("&")]]

    def get_action_space(self, name: str) -> ActionSpace:
        return self.action_spaces[name[: name.find("&")]]

    def _get_step_info(
        self, step: bool = False
    ) -> Tuple[ObsDict, RewardDict, DoneDict, InfoDict]:

        names = self.behaviors.keys()
        obs_dict: ObsDict = {}
        reward_dict: RewardDict = {}
        done_dict: DoneDict = {}
        info_dict: InfoDict = {}

        ter_obs_dict = {}
        ter_reward_dict = {}

        for name in names:
            decisions, terminals = self.unity.get_steps(name)

            behavior_specs = self.behaviors[name][0]

            n_obs_dict, n_reward_dict, n_done_dict = process_decisions(
                decisions, name, behavior_specs
            )
            n_ter_obs_dict, n_ter_reward_dict, n_ter_done_dict = process_decisions(
                terminals, name, behavior_specs
            )

            for key, value in n_ter_done_dict.items():
                n_done_dict[key] = value

            obs_dict.update(n_obs_dict)
            # breakpoint()
            reward_dict.update(n_reward_dict)
            done_dict.update(n_done_dict)

            ter_obs_dict.update(n_ter_obs_dict)
            ter_reward_dict.update(n_ter_reward_dict)

        if len(reward_dict) < len(obs_dict):
            raise ValueError("This shouldn't happen, check it")
        # done_dict["__all__"] = all(done_dict.values())

        # info_dict["final_obs"] = ter_obs_dict
        # info_dict["final_rewards"] = ter_reward_dict

        for agent_id, done in done_dict.items():
            if done:
                # TODO: For some reason this is the thing that works, idk why I don't need to update the observations
                reward_dict[agent_id] = ter_reward_dict[agent_id]
                # obs_dict[agent_id] = ter_obs_dict[agent_id]

        stats = self.stats_channel.parse_info(clear=step)
        # stats = parse_side_message(self.stats_channel.last_msg)
        for key in stats:
            info_dict[key if key.startswith("e_") else "m_" + key] = stats[key]

        return obs_dict, reward_dict, done_dict, info_dict

    def step(
        self, action: ActionDict
    ) -> Tuple[ObsDict, RewardDict, DoneDict, InfoDict]:

        for name in self.behaviors.keys():
            decisions, terminals = self.unity.get_steps(name)
            action_shape = self.behaviors[name].action_spec.continuous_size
            dec_ids = list(decisions.agent_id)

            all_actions = []
            for id_ in dec_ids:
                single_action = action.get(
                    f"{name}&id={id_}",  # Get the appropriate action
                    Action(continuous=np.zeros(action_shape)),  # Default value
                )

                cont_action = single_action.map(np.asarray).continuous.ravel()

                all_actions.append(cont_action)

            # all_actions = np.array([action.get(f"{name}&id={id_}", np.zeros(action_shape)).ravel()
            #                         for id_ in dec_ids])
            #
            if len(all_actions) == 0:
                all_actions = np.zeros((0, action_shape))
            else:
                all_actions = np.array(all_actions)

            self.unity.set_actions(name, ActionTuple(continuous=all_actions))

        # The terminal step handling has been removed as episodes are only reset from here

        self.unity.step()
        obs_dict, reward_dict, done_dict, info_dict = self._get_step_info(step=True)

        if len(reward_dict) < len(obs_dict):
            raise ValueError("Old debugging code, this shouldn't happen")

        return obs_dict, reward_dict, done_dict, info_dict

    def reset(self, **kwargs) -> ObsDict:

        self._process_parameters(kwargs)

        self.unity.reset()

        behaviors = dict(self.unity.behavior_specs)
        self.behaviors = {key: value for key, value in behaviors.items()}

        obs_dict, _, _, _ = self._get_step_info(step=True)
        if len(obs_dict) == 0:
            # Dealing with some terminal steps due to reducing the number of agents
            self.unity.step()
            obs_dict, _, _, _ = self._get_step_info(step=True)

        self.active_agents = list(obs_dict.keys())
        # if len(self.active_agents) == 0:
        #     self.reset(mode, num_agents)

        return obs_dict

    def _process_parameters(self, params: dict[str, Any]):
        for key, value in params.items():
            if isinstance(value, str):
                self.string_channel.send_string(key, value)
            elif isinstance(value, float) or isinstance(value, int):
                self.param_channel.set_float_parameter(key, float(value))

    @property
    def current_obs(self) -> ObsDict:
        obs_dict, _, _, info_dict = self._get_step_info()
        return obs_dict

    @property
    def current_info(self) -> InfoDict:
        _, _, _, info_dict = self._get_step_info()
        return info_dict

    def close(self):
        if not self._closed:
            if self.virtual_display:
                self.virtual_display.stop()
            try:
                self.unity.close()
                self._closed = True
            except UnityEnvironmentException:
                print("Trying to close Unity environment, but it was already closed")

    def render(self, mode="rgb_array") -> Optional[Union[np.ndarray, Image]]:
        if self.virtual_display:
            img = self.virtual_display.grab()
            if mode == "rgb_array":
                return np.array(img)
            else:
                return img
        else:
            return None

    def set_timescale(self, timescale: float = 100.0):
        self.engine_channel.set_configuration_parameters(time_scale=timescale)

    @classmethod
    def get_env_creator(cls, *args, **kwargs):
        def _inner():
            env = cls(*args, **kwargs)
            env.engine_channel.set_configuration_parameters(time_scale=100)
            return env

        return _inner

    @classmethod
    def get_venv(
        cls,
        workers: int = 8,
        file_name: Optional[str] = None,
        base_worker_id: Optional[int] = None,
        *args,
        **kwargs,
    ) -> SubprocVecEnv:
        if base_worker_id is None:
            base_worker_id = find_free_worker(500)
        # print(f"Using worker id {worker_id}")
        venv = SubprocVecEnv(
            [
                cls.get_env_creator(
                    file_name=file_name,
                    seed=i,
                    worker_id=base_worker_id + i,
                    *args,
                    **kwargs,
                )
                for i in range(workers)
            ]
        )
        return venv
