import numpy as np
import torch
from coltra.discounting import discount_experience, _discount_bgae, convert_params, get_beta_vector


def test_convert():
    assert np.allclose(convert_params(0.5, 0), (0.5, np.inf))
    assert np.allclose(convert_params(0.9, 0.5), (9*2, 2))
    assert np.allclose(convert_params(0.99, 0.5), (99*2, 2))
    assert np.allclose(convert_params(0.9, 1), (9, 1))


def test_beta_vector():
    # Exponential
    Γ = get_beta_vector(T=100, α=0.9, β=np.inf)

    assert Γ.shape == (100,)
    assert np.allclose(Γ, np.array([0.9**t for t in range(100)]))

    # Hyperbolic
    Γ = get_beta_vector(T=100, α=0.9, β=1)

    assert Γ.shape == (100,)
    assert np.allclose(Γ, np.array([1 / (1 + (1/0.9) * t) for t in range(100)]))

    # Some intermediate values
    Γ = get_beta_vector(T=100, α=0.9, β=2.)
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
    dones = torch.tensor([False if (t+1) % 5 else True for t in range(30)])

    returns, advantages = discount_experience(rewards, values, dones, 0.99, 0., 1.)

    assert isinstance(returns, torch.Tensor)
    assert isinstance(advantages, torch.Tensor)
    assert returns.shape == (30,)
    assert advantages.shape == (30,)
    assert torch.allclose(returns, advantages + values)

    rewards = torch.randn(1000)
    values = torch.randn(1000)
    dones = torch.tensor([False if (t+1) % 500 else True for t in range(1000)])

    returns, advantages = discount_experience(rewards, values, dones, 0.99, 0.5, 0.95)

    assert isinstance(returns, torch.Tensor)
    assert isinstance(advantages, torch.Tensor)
    assert returns.shape == (1000,)
    assert advantages.shape == (1000,)
    assert torch.allclose(returns, advantages + values)
    # assert torch.allclose(returns[:500], returns[500:])
    # assert torch.allclose(advantages[:500], advantages[500:])

    rewards = torch.ones(2000)
    values = torch.zeros(2000)
    dones = torch.tensor([False if (t+1) % 1000 else True for t in range(2000)])

    returns, advantages = discount_experience(rewards, values, dones, 0.99, 1.0, 0.95)

    assert isinstance(returns, torch.Tensor)
    assert isinstance(advantages, torch.Tensor)
    assert returns.shape == (2000,)
    assert advantages.shape == (2000,)
    assert torch.allclose(returns, advantages + values)
    assert torch.allclose(returns[:1000], returns[1000:])
    assert torch.allclose(advantages[:1000], advantages[1000:])


# def test_episode_lens():
#     dones = ...
#
# def test_episode_rewards():
#     rewards = torch.cat([torch.zeros(10), torch.zeros(10) + 1, torch.zeros(10) + 2])
#     dones = torch.tensor([...])