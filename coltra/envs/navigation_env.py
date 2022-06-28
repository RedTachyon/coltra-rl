# # WIP
# import gym
# import jax.numpy as jnp
# import jax.random
# import numpy as np
#
# from coltra import Action
# from coltra.envs import MultiAgentEnv
#
#
# @jax.jit
# def update_state(pos: jnp.ndarray, vel: jnp.ndarray, dt: float) -> jnp.ndarray:
#     pos += dt * vel
#     return pos
#
#
# @jax.jit
# def get_energy(vel: jnp.ndarray) -> jnp.ndarray:
#     vel_mag = jnp.linalg.norm(vel, axis=-1)
#     return 2.23 + 1.26 * vel_mag
#
#
# @jax.jit
# def stack_data(data: dict) -> jnp.ndarray:
#     return jnp.stack([data[k] for k in data.keys()], axis=0)
#
#
# @jax.jit
# def unstack_data(data: jnp.ndarray) -> dict:
#     return {f"agent{i}": data[i] for i, _ in enumerate(data)}
#
#
# def normalize_actions(actions: jnp.ndarray) -> jnp.ndarray:
#     return actions / jnp.linalg.norm(actions, axis=-1, keepdims=True)
#
#
# class NavigationEnv(MultiAgentEnv):
#     def __init__(
#         self, num_agents: int, time_limit: int, env_scale: float = 10.0, dt: float = 0.1
#     ):
#         super().__init__()
#         self.num_agents = num_agents
#         self.time_limit = time_limit
#         self.env_scale = env_scale
#         self.dt = dt
#
#         self.observation_space = gym.spaces.Box(-1, 1, shape=(4,))
#         self.action_space = gym.spaces.Box(-np.inf, np.inf, shape=(2,))
#
#         self.pos = None
#         self.goals = None
#
#         self.prng_key = None
#
#     def reset(self, **kwargs):
#         scale = kwargs.get("scale", self.env_scale)
#         seed = kwargs.get("seed", None)
#         if seed is None and self.prng_key is None:
#             seed = np.random.randint(0, 2**32)
#             self.prng_key = jax.random.PRNGKey(seed)
#         elif seed is None:
#             self.prng_key, subkey = jax.random.split(self.prng_key, 2)
#         else:
#             self.prng_key = jax.random.PRNGKey(seed)
#
#         pos_key, goal_key = jax.random.split(self.prng_key, 2)
#         self.pos = jax.random.uniform(pos_key, shape=(self.num_agents, 2)) * scale
#         self.goals = jax.random.uniform(goal_key, shape=(self.num_agents, 2)) * scale
#
#         return self.get_obs()
#
#     def step(self, actions: dict[str, Action]):
#         actions = {k: v.continuous for k, v in actions.items()}
#         actions = stack_data(actions)
#         self.pos = update_state(self.pos, actions, self.dt)
#         energy = get_energy(actions)
#         return self.get_obs(), np.asarray(energy), False, {}
#
#     def get_obs(self) -> dict[str, np.ndarray]:
#         obs = jnp.concatenate([self.pos, self.goals], axis=1)
#         return {f"agent{i}": np.asarray(obs[i]) for i in range(self.num_agents)}
