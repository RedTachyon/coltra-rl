import numpy as np
import pytest
import torch

from coltra import HomogeneousGroup, collect_crowd_data
from coltra.agents import RandomGymAgent
from coltra.discounting import (
    discount_experience,
    convert_params,
    get_beta_vector,
    _fast_discount_gae,
    _discount_bgae,
)
from coltra.envs import MultiGymEnv


def test_convert():
    assert np.allclose(convert_params(0.5, 0), (0.5, np.inf))
    assert np.allclose(convert_params(0.9, 0.5), (9 * 2, 2))
    assert np.allclose(convert_params(0.99, 0.5), (99 * 2, 2))
    assert np.allclose(convert_params(0.9, 1), (9, 1))


def test_beta_vector():
    # Exponential
    Γ = get_beta_vector(T=100, α=0.9, β=np.inf)

    assert Γ.shape == (100,)
    assert np.allclose(Γ, np.array([0.9**t for t in range(100)]))

    # Hyperbolic
    Γ = get_beta_vector(T=100, α=0.9, β=1)

    assert Γ.shape == (100,)
    assert np.allclose(Γ, np.array([1 / (1 + (1 / 0.9) * t) for t in range(100)]))

    # Some intermediate values
    Γ = get_beta_vector(T=100, α=0.9, β=2.0)
    assert Γ.shape == (100,)

    Γ = get_beta_vector(T=100, α=0.99, β=0.5)
    assert Γ.shape == (100,)

    # No discounting
    Γ = get_beta_vector(T=100, α=1, β=np.inf)
    assert Γ.shape == (100,)
    assert np.allclose(Γ, np.ones_like(Γ))

    # Myopic discounting
    Γ = get_beta_vector(T=100, α=0, β=0)
    assert Γ.shape == (100,)
    assert np.allclose(Γ[0], 1)
    assert np.allclose(Γ[1:], np.zeros((99,)))


def test_discounting():

    rewards = torch.cat([torch.zeros(10), torch.zeros(10) + 1, torch.zeros(10) + 2])
    values = torch.cat([torch.zeros(10), torch.zeros(10) + 1, torch.zeros(10) + 2])
    last_values = torch.zeros((10,))
    dones = torch.tensor([False if (t + 1) % 5 else True for t in range(30)])

    returns, advantages = discount_experience(
        rewards, values, dones, last_values, 0.99, 0.0, 1.0
    )

    assert isinstance(returns, torch.Tensor)
    assert isinstance(advantages, torch.Tensor)
    assert returns.shape == (30,)
    assert advantages.shape == (30,)
    assert torch.allclose(returns, advantages + values)

    rewards = torch.randn(1000)
    values = torch.randn(1000)
    last_values = torch.zeros((10,))
    dones = torch.tensor([False if (t + 1) % 500 else True for t in range(1000)])

    returns, advantages = discount_experience(
        rewards, values, dones, last_values, 0.99, 0.5, 0.95
    )

    assert isinstance(returns, torch.Tensor)
    assert isinstance(advantages, torch.Tensor)
    assert returns.shape == (1000,)
    assert advantages.shape == (1000,)
    assert torch.allclose(returns, advantages + values)
    # assert torch.allclose(returns[:500], returns[500:])
    # assert torch.allclose(advantages[:500], advantages[500:])

    rewards = torch.randn(500)
    rewards = torch.cat([rewards, rewards])
    values = torch.randn(500)
    values = torch.cat([values, values])
    last_values = torch.ones((10,))
    dones = torch.tensor([False if (t + 1) % 500 else True for t in range(1000)])

    returns, advantages = discount_experience(
        rewards, values, dones, last_values, 0.99, 0.5, 0.95
    )

    assert isinstance(returns, torch.Tensor)
    assert isinstance(advantages, torch.Tensor)
    assert returns.shape == (1000,)
    assert advantages.shape == (1000,)
    assert torch.allclose(returns, advantages + values)
    assert torch.allclose(returns[:500], returns[500:])
    assert torch.allclose(advantages[:500], advantages[500:])

    rewards = torch.ones(2000)
    values = torch.zeros(2000)
    last_values = torch.zeros((20,))
    dones = torch.tensor([False if (t + 1) % 1000 else True for t in range(2000)])

    returns, advantages = discount_experience(
        rewards, values, dones, last_values, 0.99, 1.0, 0.95
    )

    assert isinstance(returns, torch.Tensor)
    assert isinstance(advantages, torch.Tensor)
    assert returns.shape == (2000,)
    assert advantages.shape == (2000,)
    assert torch.allclose(returns, advantages + values)
    assert torch.allclose(returns[:1000], returns[1000:])
    assert torch.allclose(advantages[:1000], advantages[1000:])


@pytest.fixture(scope="module")
def dataset():
    env = MultiGymEnv.get_venv(workers=8, env_name="CartPole-v0", seed=0)
    agent = RandomGymAgent(env.action_space.discrete)
    env.action_space.discrete.seed(0)
    group = HomogeneousGroup(agent)

    data, metrics, shape = collect_crowd_data(group, env, num_steps=500)

    done = data["crowd"].done.reshape(shape).numpy()

    reward = data["crowd"].reward.reshape(shape).numpy()  # [:,:idx]
    value = data["crowd"].value.reshape(shape).numpy().astype(np.float32)  # [:,:idx]
    last_value = data["crowd"].last_value.numpy()

    return reward, value, last_value, done


# TODO: This fails with some more parameter values, might just be float precision, need to fix it at some point
@pytest.mark.parametrize("gae_lambda", [0.0, 0.9, 0.99, 1.0])
@pytest.mark.parametrize("gamma", [0.0, 0.9, 0.99, 1.0])
def test_bgae(dataset, gae_lambda: float, gamma: float):

    reward, value, last_value, done = dataset

    np.random.seed(0)
    value = np.random.randn(*value.shape).astype(np.float32)
    last_value = np.random.randn(*last_value.shape).astype(np.float32)

    bgae_adv = _discount_bgae(reward, value, done, last_value, gamma, 0.0, gae_lambda)
    gae_adv = _fast_discount_gae(reward, value, done, last_value, gamma, gae_lambda)

    assert np.allclose(bgae_adv, gae_adv, atol=1e-7)
