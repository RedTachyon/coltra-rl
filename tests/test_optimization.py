import torch
import numpy as np

from coltra.buffers import Observation
from coltra.policy_optimization import minibatches, CrowdPPOptimizer
from coltra.models.mlp_models import MLPModel
from coltra.agents import CAgent
from coltra.envs.probe_envs import ConstRewardEnv
from coltra.collectors import collect_crowd_data


def test_minibatches():
    rng = torch.manual_seed(0)

    obs = Observation(vector=torch.randn(800, 4, generator=rng))
    logprobs = torch.randn(800, generator=rng)
    values = torch.randn(800, generator=rng)

    batches = minibatches(obs, logprobs, values, batch_size=80, shuffle=False)
    count = 0
    for i, (m_obs, m_logprobs, m_values) in enumerate(batches):
        count += 1
        assert m_obs.vector.shape == (80, 4)
        assert m_logprobs.shape == (80,)
        assert m_values.shape == (80,)
        assert torch.allclose(m_obs.vector, obs[i * 80 : i * 80 + 80].vector)
        assert torch.allclose(m_logprobs, logprobs[i * 80 : i * 80 + 80])
        assert torch.allclose(m_logprobs, logprobs[i * 80 : i * 80 + 80])
    assert count == 10


def test_shuffle_minibatches():
    rng = torch.manual_seed(0)
    np_rng = np.random.default_rng(seed=0)

    obs = Observation(vector=torch.randn(800, 4, generator=rng))
    logprobs = torch.randn(800, generator=rng)
    values = torch.randn(800, generator=rng)

    batches = minibatches(
        obs, logprobs, values, batch_size=80, shuffle=True, rng=np_rng
    )
    count = 0
    for i, (m_obs, m_logprobs, m_values) in enumerate(batches):
        count += 1
        assert m_obs.vector.shape == (80, 4)
        assert m_logprobs.shape == (80,)
        assert m_values.shape == (80,)
        # Due to stochasticity of shuffling, this *could* fail if stars align and something messes up the random seed
        # But in principle, if a minibatch is not shuffled, something might be wrong
        assert not torch.allclose(m_obs.vector, obs[i * 80 : i * 80 + 80].vector)
        assert not torch.allclose(m_logprobs, logprobs[i * 80 : i * 80 + 80])
        assert not torch.allclose(m_logprobs, logprobs[i * 80 : i * 80 + 80])
    assert count == 10


def test_uneven_minibatches():
    rng = torch.manual_seed(0)

    obs = Observation(vector=torch.randn(850, 4, generator=rng))
    logprobs = torch.randn(850, generator=rng)
    values = torch.randn(850, generator=rng)

    batches = minibatches(obs, logprobs, values, batch_size=80, shuffle=False)
    count = 0
    for i, (m_obs, m_logprobs, m_values) in enumerate(batches):
        count += 1
        if count < 11:
            assert m_obs.vector.shape == (80, 4)
            assert m_logprobs.shape == (80,)
            assert m_values.shape == (80,)
            assert torch.allclose(m_obs.vector, obs[i * 80 : i * 80 + 80].vector)
            assert torch.allclose(m_logprobs, logprobs[i * 80 : i * 80 + 80])
            assert torch.allclose(m_logprobs, logprobs[i * 80 : i * 80 + 80])
        else:
            assert m_obs.vector.shape == (50, 4)
            assert m_logprobs.shape == (50,)
            assert m_values.shape == (50,)
            assert torch.allclose(m_obs.vector, obs[i * 80 : i * 80 + 80].vector)
            assert torch.allclose(m_logprobs, logprobs[i * 80 : i * 80 + 80])
            assert torch.allclose(m_logprobs, logprobs[i * 80 : i * 80 + 80])
    assert count == 11


def test_ppo_step():
    model = MLPModel({"input_size": 1, "num_actions": 2, "discrete": False})
    old_params = list([param.detach().clone() for param in model.parameters()])
    agent = CAgent(model)
    env = ConstRewardEnv(num_agents=10)

    data, metrics, shape = collect_crowd_data(
        agent, env, num_steps=100
    )  # 1000 steps total

    ppo = CrowdPPOptimizer(
        agent=agent,
        config={
            # 30 updates total
            "minibatch_size": 100,
            "ppo_epochs": 3,
            "use_gpu": torch.cuda.is_available(),
        },
    )

    data.cpu()

    metrics = ppo.train_on_data(data, shape)
    new_params = model.parameters()

    # Check that something has changed
    allclose = []
    for (p1, p2) in zip(old_params, new_params):
        allclose.append(torch.allclose(p1.cpu(), p2.cpu()))

    assert not all(allclose)
