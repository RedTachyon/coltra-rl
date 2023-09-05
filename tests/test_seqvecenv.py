import numpy as np
from coltra.envs import probe_env_classes
from coltra.collectors import collect_crowd_data
from coltra.agents import ConstantAgent
from coltra.envs.subproc_vec_env import SequentialVecEnv
from coltra.groups import HomogeneousGroup


def test_sequential_venv_reset_and_step():
    venv = SequentialVecEnv(
        [probe_env_classes[0].get_env_creator(num_agents=10) for _ in range(8)]
    )
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

    assert isinstance(info["m_stat"], np.ndarray)
    assert info["m_stat"].shape[0] == 8


def test_sequential_collect():
    venv = SequentialVecEnv(
        [probe_env_classes[0].get_env_creator(num_agents=10) for _ in range(8)]
    )
    agent = ConstantAgent(np.array([1.0]))
    agents = HomogeneousGroup(agent)

    data, stats, shape = collect_crowd_data(agents, venv, 500)
    data = data[agents.policy_name]

    assert data.obs.vector.shape == (8 * 10 * 500, 1)
    assert stats["m_stat"].shape == (500 * 8,)
