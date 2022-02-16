import multiprocessing as mp
from collections import OrderedDict
from typing import Sequence, Any, Dict, List, Callable, Union, Optional

import gym
import numpy as np

from coltra.buffers import Observation
from coltra.utils import parse_agent_name
from .base_env import VecEnv, CloudpickleWrapper
from .base_env import MultiAgentEnv


def _worker(
    remote: mp.connection.Connection,
    parent_remote: mp.connection.Connection,
    env_fn_wrapper: CloudpickleWrapper,
):
    parent_remote.close()
    env = env_fn_wrapper.var()
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == "step":
                observation, reward, done, info = env.step(data)
                # if done:
                #     # save final observation where user can get it, then reset
                #     info['terminal_observation'] = observation
                #     observation = env.reset()
                remote.send((observation, reward, done, info))
            elif cmd == "seed":
                remote.send(env.seed(data))
            elif cmd == "reset":
                observation = env.reset(**data)
                remote.send(observation)
            elif cmd == "render":
                remote.send(env.render("rgb_array"))
            elif cmd == "close":
                env.close()
                remote.close()
                break
            elif cmd == "get_spaces":
                remote.send((env.observation_space, env.action_space))
            elif cmd == "env_method":
                method = getattr(env, data[0])
                remote.send(method(*data[1], **data[2]))
            elif cmd == "get_attr":
                remote.send(getattr(env, data))
            elif cmd == "set_attr":
                remote.send(setattr(env, data[0], data[1]))
            elif cmd == "observation_space":
                remote.send(env.observation_space)
            elif cmd == "action_space":
                remote.send(env.action_space)
            elif cmd == "envs":
                remote.send(env)
            else:
                raise NotImplementedError(f"`{cmd}` is not implemented in the worker")
        except EOFError:
            break


class SubprocVecEnv(VecEnv, MultiAgentEnv):
    """
    Creates a multiprocess vectorized wrapper for multiple environments, distributing each environment to its own
    process, allowing significant speed up when the environment is computationally complex.

    For performance reasons, if your environment is not IO bound, the number of environments should not exceed the
    number of logical cores on your CPU.

    .. warning::

        Only 'forkserver' and 'spawn' start methods are thread-safe,
        which is important when TensorFlow sessions or other non thread-safe
        libraries are used in the parent (see issue #217). However, compared to
        'fork' they incur a small start-up cost and have restrictions on
        global variables. With those methods, users must wrap the code in an
        ``if __name__ == "__main__":`` block.
        For more information, see the multiprocessing documentation.

    :param env_fns: ([callable]) A list of functions that will create the environments
        (each callable returns a `Gym.Env` instance when called).
    :param start_method: (str) method used to start the subprocesses.
           Must be one of the methods returned by multiprocessing.get_all_start_methods().
           Defaults to 'forkserver' on available platforms, and 'spawn' otherwise.
    """

    def __init__(
        self,
        env_fns: list[Callable[[], MultiAgentEnv]],
        start_method: Optional[str] = None,
    ):
        self.waiting = False
        self.closed = False
        n_envs = len(env_fns)

        if start_method is None:
            # Fork is not a thread safe method (see issue #217)
            # but is more user friendly (does not require to wrap the code in
            # a `if __name__ == "__main__":`)
            forkserver_available = "forkserver" in mp.get_all_start_methods()
            start_method = "forkserver" if forkserver_available else "spawn"
        ctx = mp.get_context(start_method)

        self.remotes, self.work_remotes = zip(
            *[ctx.Pipe(duplex=True) for _ in range(n_envs)]
        )
        self.processes = []
        for work_remote, remote, env_fn in zip(
            self.work_remotes, self.remotes, env_fns
        ):
            args = (work_remote, remote, CloudpickleWrapper(env_fn))
            # daemon=True: if the main process crashes, we should not cause things to hang
            process = ctx.Process(
                target=_worker, args=args, daemon=True
            )  # pytype:disable=attribute-error
            process.start()
            self.processes.append(process)
            work_remote.close()

        self.remotes[0].send(("get_spaces", None))
        observation_space, action_space = self.remotes[0].recv()
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

    def step_async(self, actions):
        """Send the actons to the environments"""
        for i, remote in enumerate(self.remotes):
            action = {
                "&".join(k.split("&")[:-1]): a
                for k, a in actions.items()
                if int(parse_agent_name(k)["env"]) == i
            }
            remote.send(("step", action))
        # for remote, action in zip(self.remotes, actions):
        #     remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        # infos - tuple of dicts
        return (
            _gather_subproc(obs),
            _gather_subproc(rews),
            _gather_subproc(dones),
            _flatten_info(infos),
        )

    def seed(self, seed=None):
        for idx, remote in enumerate(self.remotes):
            remote.send(("seed", seed + idx))
        return [remote.recv() for remote in self.remotes]

    def reset(self, **kwargs) -> Dict[str, Observation]:
        for remote in self.remotes:
            remote.send(("reset", kwargs))
        obs = [remote.recv() for remote in self.remotes]
        return _gather_subproc(obs)

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(("close", None))
        for process in self.processes:
            process.join()
        self.closed = True

    def get_images(self) -> Sequence[np.ndarray]:
        for pipe in self.remotes:
            # gather images from subprocesses
            # `mode` will be taken into account later
            pipe.send(("render", "rgb_array"))
        imgs = [pipe.recv() for pipe in self.remotes]
        return imgs

    def render(self, **kwargs) -> np.ndarray:
        pipe = self.remotes[0]
        pipe.send(("render", "rgb_array"))
        img = pipe.recv()
        return img

    def get_attr(self, attr_name, indices=None):
        """Return attribute from vectorized environment (see base class)."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("get_attr", attr_name))
        return [remote.recv() for remote in target_remotes]

    def set_attr(self, attr_name, value, indices=None):
        """Set attribute inside vectorized environments (see base class)."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("set_attr", (attr_name, value)))
        for remote in target_remotes:
            remote.recv()

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        """Call instance methods of vectorized environments."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("env_method", (method_name, method_args, method_kwargs)))
        return [remote.recv() for remote in target_remotes]

    def get_envs(self):
        for idx, remote in enumerate(self.remotes):
            remote.send(("envs", None))
        return [remote.recv() for remote in self.remotes]

    def _get_target_remotes(self, indices):
        """
        Get the connection object needed to communicate with the wanted
        envs that are in subprocesses.

        :param indices: (None,int,Iterable) refers to indices of envs.
        :return: ([multiprocessing.Connection]) Connection object to communicate between processes.
        """
        indices = self._get_indices(indices)
        return [self.remotes[i] for i in indices]


def _gather_subproc(obs: List[Dict[str, Observation]]) -> Dict[str, Observation]:
    combined_obs = {
        f"{key}&env={i}": value
        for i, s_obs in enumerate(obs)
        for (key, value) in s_obs.items()
    }
    return combined_obs


def _flatten_scalar(values: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    keys = values[0].keys()
    return {k: np.array([v[k] for v in values]) for k in keys}


def _flatten_info(
    infos: List[Dict[str, np.ndarray]]
) -> Dict[str, Union[np.ndarray, List]]:
    all_metrics = {}

    all_keys = set([k for dictionary in infos for k in dictionary])
    for key in all_keys:
        if key.startswith("m_") or key.startswith("e_"):
            all_metrics[key] = np.concatenate(
                [info_i[key] for info_i in infos if key in info_i]
            )
        else:
            all_metrics[key] = [
                info_i[key] if key in info_i else None for info_i in infos
            ]

    return all_metrics
