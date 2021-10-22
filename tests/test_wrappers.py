import gym

from coltra.buffers import discrete
from coltra.envs import MultiGymEnv
from coltra.wrappers.env_wrappers import LastRewardWrapper


def test_reward_wrapper():
    env = MultiGymEnv.get_venv(8, env_name="CartPole-v1", wrappers=[LastRewardWrapper])

    ref_env = gym.make("CartPole-v1")

    assert env.observation_space.shape[0] == ref_env.observation_space.shape[0] + 1

    obs = env.reset()

    for agent_id in obs:
        assert obs[agent_id].vector[-1] == 0.0
        assert obs[agent_id].vector.shape == env.observation_space.shape

    for _ in range(10):
        obs, reward, done, info = env.step(
            {agent_id: discrete(env.action_space.sample()) for agent_id in obs}
        )

        for agent_id in obs:
            assert obs[agent_id].vector[-1] == reward[agent_id]
            assert obs[agent_id].vector.shape == env.observation_space.shape
