from typing import Callable, Optional
import multiprocessing as mp

from coltra.envs import MultiAgentEnv

def env_worker(pipe: mp.connection.Connection, env_fn: Callable) -> None:
    env = env_fn()
    pipe.send(env)
    pipe.close()


class VectorEnv(MultiAgentEnv):
    def __init__(self, env_fn: Callable[[], MultiAgentEnv], workers: int = 8, start_method: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.env_fn = env_fn
        self.workers = workers

        start_method = start_method or 'forkserver'

        ctx = mp.get_context(start_method if start_method in mp.get_all_start_methods() else 'spawn')

        self.pipes, self.worker_pipes = zip(*[ctx.Pipe() for _ in range(workers)])
        self.workers = [ctx.Process(target=env_worker, args=(worker_pipe,)) for worker_pipe in self.worker_pipes]

