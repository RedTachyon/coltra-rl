from typing import List, Tuple, Optional, Type

from typarse import BaseConfig


class MLPConfig(BaseConfig):
    input_size: int = 0  # Must be set
    num_actions: int = 0  # Must be set
    discrete: bool = None  # Must be set

    activation: str = "leaky_relu"
    sigma0: float = 1.0

    std_head: bool = False

    hidden_sizes: List[int] = [64, 64]

    initializer: str = "kaiming_uniform"


class OptimizerKwargs(BaseConfig):
    lr: float = 1e-4
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-7
    weight_decay: float = 0.0
    amsgrad: bool = False


class PPOConfig(BaseConfig):
    # Discounting and GAE - by default, exponential discounting at γ=0.99
    gamma: float = 0.99
    eta: float = 0.0
    gae_lambda: float = 1.0

    use_ugae: bool = False

    # PPO optimization parameters
    eps: float = 0.1
    target_kl: float = 0.03
    entropy_coeff: float = 0.001
    entropy_decay_time: float = 100.0
    min_entropy: float = 0.001
    value_coeff: float = 1.0  # Technically irrelevant
    advantage_normalization: bool = False

    # Number of gradient updates = ppo_epochs * ceil(batch_size / minibatch_size)
    ppo_epochs: int = 3
    minibatch_size: int = 8192

    use_gpu: bool = False

    optimizer: str = "adam"

    OptimizerKwargs: Type[OptimizerKwargs] = OptimizerKwargs.clone()


class TrainerConfig(BaseConfig):
    steps: int = 500
    workers: int = 8

    mode: str = "random"
    num_agents: int = 20

    tensorboard_name: Optional[str] = None
    save_freq: int = 10

    PPOConfig: Type[PPOConfig] = PPOConfig.clone()


class LeeConfig(BaseConfig):
    input_size: int = 4
    rays_input_size: int = 126

    conv_filters: int = 2


class RelationConfig(BaseConfig):
    input_size: int = 7
    num_actions: int = 2
    rel_input_size: int = 4

    vec_hidden_layers: List[int] = [32, 32]
    rel_hidden_layers: List[int] = [32, 32]
    com_hidden_layers: List[int] = [32, 32]

    activation: str = "tanh"
    initializer: str = "orthogonal"
