import socket
from sys import platform

from mlagents_envs.exception import UnityWorkerInUseException
from mlagents_envs.logging_util import set_log_level, ERROR


def is_worker_free(worker_id: int, base_port: int = 5005):
    """
    Attempts to bind to the requested communicator port, checking if it is already in use.
    Returns whether the port is free.
    """
    port = base_port + worker_id
    status = True
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    if platform == "linux" or platform == "linux2":
        # On linux, the port remains unusable for TIME_WAIT=60 seconds after closing
        # SO_REUSEADDR frees the port right after closing the environment
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        s.bind(("localhost", port))
    except OSError:
        status = False
    #         raise UnityWorkerInUseException(self.worker_id)
    finally:
        s.close()

    return status


def find_free_worker(max_value: int = 1000, step: int = 10) -> int:
    """
    Finds a free worker ID.
    """
    for worker_id in range(0, max_value, step):
        if is_worker_free(worker_id):
            return worker_id

    raise UnityWorkerInUseException("All workers are in use.")


def disable_unity_logs():
    set_log_level(ERROR)
