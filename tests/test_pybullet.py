from coltra.envs.gym_envs import MultiGymEnv, import_bullet


def test_init():
    env = MultiGymEnv.get_venv(
        8, env_name="HopperBulletEnv-v0", seed=0, import_fn=import_bullet
    )
    obs = env.reset()

    assert len(obs) == 8
