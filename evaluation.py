import os
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
from torch import Tensor


def load_weights(base_path: str,
                 base_name: str = 'weights',
                 start: int = 0,
                 end: Optional[int] = None) -> List[Dict[str, Tensor]]:
    """
    Loads a list of model weights from a certain path
    """
    base_path = os.path.join(base_path, "saved_weights")
    weight_paths = sorted(
        [
            os.path.join(base_path, fname)
            for fname in os.listdir(base_path)
            if fname.startswith(base_name)
        ],
        key=lambda x: int(x.split('_')[-1])
    )
    weight_paths = weight_paths[start:end]

    return [torch.load(path) for path in weight_paths]


# def load_agent_population(base_path: str,
#                           agent_fname: str = 'base_agent.pt',
#                           weight_fname: str = 'weights',
#                           start: int = 0,
#                           end: Optional[int] = None) -> Tuple[Agent, List[Dict[str, Tensor]], np.ndarray]:
#     """
#     Convenience function to load an agent along with its historic weights.
#     The files in an appropriate format are generated in the sampling trainer, in the tensorboard log directory.
#
#     Args:
#         base_path: path to the directory holding the saved agent and weights; usually tensorboard logdir
#         agent_fname: filename of the agent file
#         weight_fname: beginning of the weight filenames
#         start: starting weight index that should be loaded; assumes
#         end: last weight index that should be loaded
#
#     Returns:
#
#     """
#     base_agent, _, returns = load_agent(base_path=base_path, fname=agent_fname)
#     weights = load_weights(base_path=base_path, base_name=weight_fname, start=start, end=end)
#
#     return base_agent, weights, returns
