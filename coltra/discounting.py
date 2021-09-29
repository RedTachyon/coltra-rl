from typing import Tuple, List

import numpy as np
from numba import njit, NumbaPerformanceWarning
import torch
from torch import Tensor

from coltra.buffers import Reward, Value, Done

import warnings

warnings.simplefilter("ignore", category=NumbaPerformanceWarning)

# Disable njit for whatever reason
# njit = lambda x: x


@njit
def get_episode_rewards(
    rewards: np.ndarray, dones: np.ndarray, shape: Tuple[int, int]
) -> np.ndarray:
    rewards = rewards.reshape(shape)
    dones = dones.reshape(shape)
    batch_size, num_steps = shape

    returns = []
    for i in range(batch_size):
        ep_return = np.float32(0.0)
        for t in range(num_steps):
            ep_return += rewards[i, t]
            if dones[i, t]:
                returns.append(ep_return)
                ep_return = np.float32(0.0)

    returns = np.array(returns)
    return returns


@njit
def get_episode_lengths(dones: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    dones = dones.reshape(shape)
    batch_size, num_steps = shape

    lengths = []
    for i in range(batch_size):
        ep_len = 0
        for t in range(num_steps):
            ep_len += 1
            if dones[i, t]:
                lengths.append(ep_len)
                ep_len = 0

    lengths = np.array(lengths)
    return lengths


def discount_experience(
    rewards: Tensor,  # float tensor (T, )
    values: Tensor,  # float tensor (T, )
    dones: Tensor,  # boolean tensor (T, )
    last_values: Tensor,
    γ: float = 0.99,  # \in (0, 1); extreme values imply with eta = 0
    η: float = 0.0,
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
        rewards: 1D [B*T] array containing rewards obtained at a given step
        values: same as above, containing value estimates
        dones: same as above, containing last step flags
        last_values: 1D [B] array containing the final values of each agent/env
        γ: actually μ in the paper - the center of the discounting distribution, OR the exponential discount factor
        η: measure of dispersion of the distribution
        λ: GAE lambda parameter
        gamma: same as γ, for whenever you need to use ascii
        eta: same as η, see above
    """
    if not use_ugae:
        np_last_vals = last_values.detach().cpu().numpy().astype(np.float32)
        batch_size = np_last_vals.shape
        np_rewards = (
            rewards.detach()
            .cpu()
            .numpy()
            .astype(np.float32)
            .reshape(batch_size + (-1,))
        )
        np_values = (
            values.detach().cpu().numpy().astype(np.float32).reshape(batch_size + (-1,))
        )
        np_dones = (
            dones.detach().cpu().numpy().astype(np.float32).reshape(batch_size + (-1,))
        )

        advantages = _fast_discount_gae(
            np_rewards, np_values, np_dones, np_last_vals, γ, λ
        )

    else:  # UGAE
        raise NotImplementedError
        # ep_lens = get_episode_lens(dones)
        # ep_len = ep_lens[0]
        # for val in ep_lens:
        #     assert val == ep_len, "Episodes need to be of constant length for bGAE"
        #
        # np_rewards = (
        #     rewards.view((-1, ep_len)).detach().cpu().numpy().astype(np.float32)
        # )
        # np_values = values.view((-1, ep_len)).detach().cpu().numpy().astype(np.float32)
        # np_dones = dones.view((-1, ep_len)).detach().cpu().numpy()
        #
        # # rewards = rewards.cpu().numpy()
        # # values = values.detach().cpu().numpy()
        # # dones = dones.cpu().numpy()
        #
        # advantages = _discount_bgae(
        #     np_rewards, np_values, np_dones, γ, η, λ, gamma=gamma, eta=eta
        # )

    advantages = torch.as_tensor(advantages, device=values.device).reshape((-1,))
    returns = advantages + values  # A = R - V

    return returns.view(-1), advantages.view(-1)


@njit
def convert_params(μ: float, η: float) -> Tuple[float, float]:
    """
    Convert the mu-eta parametrization to alpha-beta
    """
    if η == 0:
        α = μ  # Exponential discounting
        β = np.inf
    else:
        α = μ / (η * (np.float32(1.0) - μ))
        β = 1.0 / η

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


@njit
def _discount_bgae(
    rewards: np.ndarray,  # float tensor (T, N)
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
        s_values = values[:, t + 1 :]
        old_value = values[:, t]

        Γ_v = Γ_all[: T - t]
        Γ_v1 = Γ_all[1 : T - t]
        λ_v = λ_all[: T - t]

        future_rewards = s_rewards @ (λ_v * Γ_v)
        future_values = s_values @ (np.float32(1.0 - λ) * (λ_v[:-1] * Γ_v1))

        advantage = -old_value + future_rewards + future_values
        advantages[:, t] = advantage

    return advantages


@njit
def _fast_discount_gae(
    rewards: np.ndarray,  # [B, T] shape
    values: np.ndarray,
    dones: np.ndarray,
    last_vals: np.ndarray,
    γ: float = 0.99,
    λ: float = 0.95,
):
    γ = np.float32(γ)
    λ = np.float32(λ)
    advantages = np.zeros_like(rewards)
    lastgaelam = np.zeros_like(rewards[:, 0])
    buffer_size = rewards.shape[-1]
    for t in range(buffer_size - 1, -1, -1):
        if t == buffer_size - 1:
            nextvalue = last_vals
            nextnonterminal = np.ones_like(dones[:, t]) - dones[:, t]
        else:
            nextvalue = values[:, t + 1]
            nextnonterminal = np.ones_like(dones[:, t]) - dones[:, t + 1]

        value = values[:, t]
        delta = rewards[:, t] + γ * nextvalue * nextnonterminal - value
        lastgaelam = delta + γ * λ * lastgaelam * nextnonterminal

        advantages[:, t] = lastgaelam

    return advantages
