from typing import Tuple, List

import numba
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

    np_last_vals = last_values.detach().cpu().numpy().astype(np.float32)
    batch_size = np_last_vals.shape
    np_rewards = (
        rewards.detach().cpu().numpy().astype(np.float32).reshape(batch_size + (-1,))
    )
    np_values = (
        values.detach().cpu().numpy().astype(np.float32).reshape(batch_size + (-1,))
    )
    np_dones = (
        dones.detach().cpu().numpy().astype(np.float32).reshape(batch_size + (-1,))
    )
    if not use_ugae:
        advantages = _fast_discount_gae(
            np_rewards, np_values, np_dones, np_last_vals, γ, λ
        )
    else:  # UGAE
        advantages = _discount_bgae(
            np_rewards, np_values, np_dones, np_last_vals, γ, η, λ
        )

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
def _bgae_one_episode(
    rewards: np.ndarray,  # (T,)
    values: np.ndarray,  # (T,)
    last_value: float,
    α: float,
    β: float,
    λ: float,
    is_final: bool = False,
) -> np.ndarray:
    """
    Compute the discounted advantage for one episode.
    """
    # Compute the discounted advantage
    T = rewards.shape[0]
    Γ = get_beta_vector(T + 1, α, β)
    lambdas = np.array([λ**l for l in range(T)], dtype=np.float32)
    advantages = np.empty_like(rewards, dtype=np.float32)

    values = np.append(values, np.float32(last_value))

    # Note to self: this is a weird trick to make the final episode returns match up with reference implementations.
    # I don't entirely understand it at the moment, but I guess it works?
    one_minus_lambda = np.array([1 - λ for _ in range(T)], dtype=np.float32)
    if is_final:
        one_minus_lambda[-1] = 1.0

    for t in range(T):
        t_left = T - t
        reward_term = (lambdas[:t_left] * Γ[:t_left]) @ rewards[t : t + t_left]
        value_term = (
            one_minus_lambda
            * (lambdas[:t_left] * Γ[1 : t_left + 1])
            @ values[t + 1 : t + t_left + 1]
        )
        advantages[t] = -values[t] + reward_term + value_term
        one_minus_lambda = one_minus_lambda[1:]

    return advantages


@njit
def _discount_bgae(
    rewards: np.ndarray,  # float tensor (N, T)
    values: np.ndarray,  # float tensor (N, T)
    dones: np.ndarray,  # boolean tensor (N, T)
    last_values: np.ndarray,  # float tensor (N,)
    γ: float = 0.99,
    η: float = 0.95,
    λ: float = 0.95,
):
    # TODO: this is probably broken by making it match the normal discounting
    N = rewards.shape[0]
    T = rewards.shape[1]
    advantages = np.empty_like(rewards, dtype=np.float32)

    α, β = convert_params(γ, η)

    for i in range(N):
        rewards_i = rewards[i]
        values_i = values[i]
        dones_i = dones[i]

        # TODO: this is messed up, at least with the new env - maybe also with new crowds?
        split = np.where(dones_i)[0] + 1
        reward_parts = np.split(rewards_i, split)
        value_parts = np.split(values_i, split)

        adv_parts = numba.typed.List()

        for j, (rew_part, val_part) in enumerate(zip(reward_parts, value_parts)):
            if len(rew_part) == 0:
                continue
            if j == len(reward_parts) - 1:
                last_val = last_values[i]
                is_final = True
            else:
                last_val = 0.0
                is_final = False

            adv_part = _bgae_one_episode(
                rew_part.astype(np.float32),
                val_part.astype(np.float32),
                last_val,
                α,
                β,
                λ,
                is_final,
            )
            adv_parts.append(adv_part)

        idx = 0
        for j, adv_part in enumerate(adv_parts):
            for k, v in enumerate(adv_part):
                advantages[i, idx] = v
                idx += 1

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
    # TODO: this is probably broken with my data collection semantic
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
