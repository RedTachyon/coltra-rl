from typing import Tuple, List

import numpy as np
from numba import njit, NumbaPerformanceWarning
import torch
from torch import Tensor

from coltra.buffers import Reward, Value, Done

import warnings

warnings.simplefilter('ignore', category=NumbaPerformanceWarning)

# Disable njit for whatever reason
# njit = lambda x: x


def get_episode_lens(dones: Tensor) -> List[int]:
    """
    Based on the recorded done values, returns the length of each episode in a batch.
    Args:
        dones: boolean tensor which values indicate terminal episodes

    Returns:
        tuple of episode lengths
    """
    episode_indices = dones.to(torch.int).cumsum(dim=0)[:-1]
    episode_indices = torch.cat([torch.tensor([0]), episode_indices])  # [0, 0, 0, ..., 1, 1, ..., 2, ..., ...]

    ep_ids, ep_lens_tensor = torch.unique(episode_indices, return_counts=True)
    ep_lens = tuple(ep_lens_tensor.cpu().numpy())

    return ep_lens


def get_episode_rewards(rewards: Tensor, dones: Tensor) -> Tensor:
    """Computes the total reward in each episode in a data batch"""
    ep_lens = get_episode_lens(dones)

    ep_rewards = torch.tensor([torch.sum(rewards) for rewards in torch.split(rewards, ep_lens)])

    return ep_rewards


@njit
def _discount_bgae(rewards: np.ndarray,  # float tensor (T, N)
                   values: np.ndarray,  # float tensor (T, N)
                   dones: np.array,  # boolean tensor (T, N)
                   γ: float,  # \in (0, 1); extreme values imply with eta = 0
                   η: float = 0,
                   λ: float = 0.95,  # \in [0, 1]; possible [0, \infty)
                   *,
                   gamma: float = None,
                   eta: float = None
                   ) -> np.ndarray:
    """
    A numpy/numba-based CPU-optimized computation. Wrapped by discount_bgae for a Tensor interface
    """
    if gamma is not None:
        γ = gamma
    if eta is not None:
        η = eta

    T = rewards.shape[1]
    assert T == values.shape[1]

    α, β = convert_params(γ, η)

    advantages = np.empty_like(rewards, dtype=np.float32)

    Γ_all = get_beta_vector(T + 1, α, β)

    λ_all = np.array([λ ** i for i in range(T)], dtype=np.float32)

    for t in range(T):
        s_rewards = rewards[:, t:]
        s_values = values[:, t + 1:]
        old_value = values[:, t]

        Γ_v = Γ_all[:T-t]
        Γ_v1 = Γ_all[1:T-t]
        λ_v = λ_all[:T-t]

        future_rewards = s_rewards @ (λ_v * Γ_v)
        future_values = s_values @ (np.float32(1.-λ) * (λ_v[:-1] * Γ_v1))

        advantage = -old_value + future_rewards + future_values
        advantages[:, t] = advantage

    return advantages


def discount_experience(rewards: Tensor,  # float tensor (T, )
                        values: Tensor,  # float tensor (T, )
                        dones: Tensor,  # boolean tensor (T, )
                        γ: float = 0.99,  # \in (0, 1); extreme values imply with eta = 0
                        η: float = 0.,
                        λ: float = 0.95,  # \in [0, 1]; possible [0, \infty)
                        use_ugae: bool = False,
                        *,
                        gamma: float = None,
                        eta: float = None
                        ) -> Tuple[Tensor, Tensor]:
    """
    Performs discounting and advantage estimation using the βGAE algorithm described in (Kwiatkowski et al., 2021).

    The general structure of all tensors is that they contain end-to-end concatenated episodes,
    with `done == True` corresponding to the last step of an episode.


    Args:
        rewards: 1D array containing rewards obtained at a given step
        values: same as above, containing value estimates
        dones: same as above, containing last step flags
        γ: actually μ in the paper - the center of the discounting distribution, OR the exponential discount factor
        η: measure of dispersion of the distribution
        λ: GAE lambda parameter
        gamma: same as γ, for whenever you need to use ascii
        eta: same as η, see above
    """
    if not use_ugae:
        np_rewards = rewards.detach().cpu().numpy().astype(np.float32)
        np_values = values.detach().cpu().numpy().astype(np.float32)
        np_dones = dones.detach().cpu().numpy()

        advantages = _fast_discount_gae(np_rewards, np_values, np_dones, γ, λ)

    else:
        ep_lens = get_episode_lens(dones)
        ep_len = ep_lens[0]
        for val in ep_lens:
            assert val == ep_len, "Episodes need to be of constant length for bGAE"

        # breakpoint()

        np_rewards = rewards.view((-1, ep_len)).detach().cpu().numpy().astype(np.float32)
        np_values = values.view((-1, ep_len)).detach().cpu().numpy().astype(np.float32)
        np_dones = dones.view((-1, ep_len)).detach().cpu().numpy()

        # rewards = rewards.cpu().numpy()
        # values = values.detach().cpu().numpy()
        # dones = dones.cpu().numpy()

        advantages = _discount_bgae(np_rewards, np_values, np_dones, γ, η, λ, gamma=gamma, eta=eta)

    advantages = torch.as_tensor(advantages.ravel(), device=values.device)
    returns = advantages + values  # A = R - V

    return returns, advantages


@njit
def convert_params(μ: float, η: float) -> Tuple[float, float]:
    """
    Convert the mu-eta parametrization to alpha-beta
    """
    if η == 0:
        α = μ  # Exponential discounting
        β = np.inf
    else:
        α = μ / (η * (np.float32(1.) - μ))
        β = 1. / η

    return α, β


@njit
def get_beta_vector(T: int, α: float, β: float) -> np.ndarray:
    discount = np.empty((T,), dtype=np.float32)

    current_discount = 1
    for t in range(T):
        discount[t] = current_discount
        if β < np.inf and α > 0:
            factor = (α + t) / (α + β + t)
        else:
            factor = α
        current_discount *= factor

    return discount


# def discount_gae(rewards: Tensor, values: Tensor, dones: Tensor, γ: float = 0.99, λ: float = 0.95):
#     advantages = torch.zeros_like(rewards)
#     lastgaelam = 0
#     buffer_size = rewards.shape[0]
#
#     for t in reversed(range(buffer_size)):
#         if t == buffer_size - 1 or dones[t]:
#             nextvalue = 0.
#             lastgaelam = 0
#         else:
#             nextvalue = values[t + 1]
#
#         value = values[t]
#         delta = rewards[t] + γ * nextvalue - value
#         lastgaelam = delta + γ * λ * lastgaelam
#
#         advantages[t] = lastgaelam
#
#     returns = advantages + values
#
#     return returns, advantages


@njit
def _fast_discount_gae(rewards: np.ndarray, values: np.ndarray, dones: np.ndarray, γ: float = 0.99, λ: float = 0.95):
    advantages = np.zeros_like(rewards)
    lastgaelam = 0
    buffer_size = rewards.shape[0]

    for t in range(buffer_size - 1, -1, -1):
        if t == buffer_size - 1 or dones[t]:
            nextvalue = 0.
            lastgaelam = 0
        else:
            nextvalue = values[t + 1]

        value = values[t]
        delta = rewards[t] + γ * nextvalue - value
        lastgaelam = delta + γ * λ * lastgaelam

        advantages[t] = lastgaelam

    return advantages
