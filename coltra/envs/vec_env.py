# Massively deprecated, is slow af

# import multiprocessing as mp
# import sys
# from queue import Queue
# from threading import Thread
# from typing import Callable, Any, Union
#
# import numpy as np
#
# from coltra.buffers import Observation, Action
# from coltra.envs import MultiAgentEnv
# from coltra.envs.base_env import VecEnv
# from coltra.utils import parse_agent_name
#
#
# def _worker(in_q: Queue, out_q: Queue, env_fn: Callable[[], MultiAgentEnv]):
#     # Setup
#     env = env_fn()
#     while True:
#         command, data = in_q.get()
#         if command == 'step':
#             out_q.put(env.step(data))
#         elif command == 'reset':
#             out_q.put(env.reset(**data))
#         elif command == 'close':
#             env.close()
#             break
#         else:
#             raise ValueError('Unknown command: {}'.format(command))
#
#
# class ThreadVectorEnv(MultiAgentEnv):
#     """
#     Vectorized environment that runs multiple environments in parallel in a
#     thread.
#
#     Parameters
#     ----------
#     env_fn : Callable[[], MultiAgentEnv]
#         Function that creates the environments.
#     num_envs : int
#         Number of environments to run in parallel.
#     """
#     def __init__(self, env_fns: list[Callable[[], MultiAgentEnv]], **kwargs):
#         super().__init__(**kwargs)
#         self.num_envs = len(env_fns)
#         self.in_qs = []
#         self.out_qs = []
#         self.workers = []
#         for i in range(self.num_envs):
#             in_q, out_q = Queue(), Queue()
#             t = Thread(target=_worker, args=(in_q, out_q, env_fns[i]))
#             self.workers.append(t)
#             self.in_qs.append(in_q)
#             self.out_qs.append(out_q)
#             t.start()
#
#         self.waiting = False
#
#     def step_async(self, actions: dict[str, Action]) -> None:
#         for i in range(self.num_envs):
#             action = {
#                 "&".join(k.split("&")[:-1]): a
#                 for k, a in actions.items()
#                 if int(parse_agent_name(k)["env"]) == i
#             }
#             self.in_qs[i].put(('step', action))
#
#     def step_wait(self):
#         results = [q.get() for q in self.out_qs]
#         self.waiting = False
#         obs, rews, dones, infos = zip(*results)
#         # infos - tuple of dicts
#         return (
#             _gather_subproc(obs),
#             _gather_subproc(rews),
#             _gather_subproc(dones),
#             _flatten_info(infos),
#         )
#
#     def step(self, actions: Any) -> Any:
#         self.step_async(actions)
#         return self.step_wait()
#
#     def close(self):
#         for i in range(self.num_envs):
#             self.in_qs[i].put(('close', None))
#
#     def reset(self, **kwargs):
#         for i in range(self.num_envs):
#             self.in_qs[i].put(('reset', kwargs))
#
#         obs = [self.out_qs[i].get() for i in range(self.num_envs)]
#         return _gather_subproc(obs)
#
#     @classmethod
#     def get_venv(cls, workers: int = 8, *args, **kwargs) -> VecEnv:
#         raise ValueError("Can't get a vector env of a vector env")
#
#     def render(self, mode="rgb_array"):
#         pass
#
# def _gather_subproc(obs: list[dict[str, Observation]]) -> dict[str, Observation]:
#     combined_obs = {
#         f"{key}&env={i}": value
#         for i, s_obs in enumerate(obs)
#         for (key, value) in s_obs.items()
#     }
#     return combined_obs
#
#
# def _flatten_info(
#         infos: list[dict[str, np.ndarray]]
# ) -> dict[str, Union[np.ndarray, list]]:
#     all_metrics = {}
#
#     all_keys = set([k for dictionary in infos for k in dictionary])
#     for key in all_keys:
#         if key.startswith("m_") or key.startswith("e_"):
#             all_metrics[key] = np.concatenate(
#                 [info_i[key] for info_i in infos if key in info_i]
#             )
#         else:
#             all_metrics[key] = [
#                 info_i[key] if key in info_i else None for info_i in infos
#             ]
#
#     return all_metrics
