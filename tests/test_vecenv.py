import numpy as np

from coltra.envs import probe_env_classes
from coltra.collectors import collect_crowd_data
from coltra.agents import ConstantAgent


def test_venv():

    venv = probe_env_classes[0].get_venv(workers=8, num_agents=10)
    obs = venv.reset()
    assert len(obs) == 80
    for env_idx in range(8):
        for agent_idx in range(10):
            assert f"Agent{agent_idx}&env={env_idx}" in obs

    obs, reward, done, info = venv.step({})

    for env_idx in range(8):
        for agent_idx in range(10):
            name = f"Agent{agent_idx}&env={env_idx}"
            assert name in obs
            assert name in reward
            assert name in done

    assert info["m_stat"].shape[0] == 8


def test_collect():
    venv = probe_env_classes[0].get_venv(workers=8, num_agents=10)
    agent = ConstantAgent(np.array([1.0]))

    data, stats = collect_crowd_data(agent, venv, 500)

    assert data.obs.vector.shape == (8 * 10 * 500, 1)
    assert stats["stat"].shape == (500, 8)
