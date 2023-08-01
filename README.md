<!-- # Coltra RL -->

![coltra logo](https://user-images.githubusercontent.com/19414946/139559727-d71caab7-1467-47a5-ac82-acdb9062e85f.png)

Figured I can finally open-source this. 
Coltra, a portmanteau of **Col**lect and **tra**in is the RL framework I've been developing for my PhD work due to a frustration with all the other existing libraries.



At the time of writing, it only contains an implementation of PPO, although I intend to change that soon. 
And if my initial designs were correct, that should prove to be quite easy. Note: the current code
is tightly connected with my thesis work. After I defend, I might start decoupling it, and we'll see what happens then.

My main philosophy of coltra is that it should be easy to modify, and easy to access literally any detail of the RL algorithm that you might want. 
For that reason, I expect that many potential users would even create their own forks, adapting the code to their own needs.

General note about terminology: the context of this project is crowd simulation, so the word "crowd" will pop up sometimes.
You can assume it basically just refers to "homogeneous multiagent something with parameter sharing"

## Another RL framework? Why?

Simple answer - because I wasn't able to use any of the existing ones. Stable Baselines 3 only barely supports multiagent scenarios,
and is only barely hackable. RLlib is super fast, but a nightmare to modify in any way that deviates from the norm. 
CleanRL is very simple, but components are not reusable at all.

Coltra can be thought of as a linear interpolation between CleanRL and SB3, with a focus on multiagent environments.

## Quickstart

Proper docs are yet to be written, but here's an outline of how to use this library.

### Installation

The library was initially written on Python 3.8, and then ported to 3.9. 
We have no guarantees that it will work on any earlier versions, although it should be easy to make that happen.

The procedure is the usual

```shell
git clone https://github.com/redtachyon/coltra-rl
cd coltra-rl
pip install -r requirements.txt
pip install -e .
```

We particularly invite people to make their forks of the library to implement whatever crazy ideas.

### At a glance

```python
from tqdm import trange

from coltra.models import MLPModel
from coltra.agents import DAgent
from coltra.groups import HomogeneousGroup

from coltra.envs import MultiGymEnv
from coltra.policy_optimization import CrowdPPOptimizer
from coltra.collectors import collect_crowd_data

if __name__ == '__main__':
    env = MultiGymEnv.get_venv(8, env_name="CartPole-v1")

    agents = HomogeneousGroup(  # Parameter-shared group agent
        DAgent(  # Individual Discrete Agent
            MLPModel(  # Policy and Value neural network
                {
                    "input_size": env.observation_space.shape[0],
                    "num_actions": env.action_space.n,
                    "discrete": True
                },
                action_space=env.action_space
            )
        )
    )

    ppo = CrowdPPOptimizer(  # PPO optimizer with full parameter sharing
        agents=agents,  # We're optimizing the agents in the group
        config={}  # Default config
    )

    for _ in trange(10):
        # Collect a batch of data using the current policy
        data_batch, collector_metrics, data_shape = collect_crowd_data(agents=agents, env=env, num_steps=100)

        # Train the current policy on the data, using PPO
        metrics = ppo.train_on_data(data_dict=data_batch, shape=data_shape)

    print(metrics)
    env.close()


```


## Usage

Here we describe the main abstractions and how to actually use this library.

### Configs
One basic unintuitive thing might be the usage of `typarse`. Check it out on [GitHub](https://github.com/RedTachyon/typarse)
to know more, but it's a tool to generate argparsers and configs based on type hints. 
In particular, you can do something like this

```python
from typarse import BaseConfig
from typing import List

class MLPConfig(BaseConfig):
    input_size: int = 0  # Must be set
    num_actions: int = 0  # Must be set
    discrete: bool = None  # Must be set

    activation: str = "leaky_relu"
    sigma0: float = 0.5

    std_head: bool = True

    hidden_sizes: List[int] = [64, 64]

    initializer: str = "kaiming_uniform"

Config: MLPConfig = MLPConfig.clone()

config = {  # Read externally, e.g. from a yaml
    "input_size": 5,
    "num_actions": 2,
    "discrete": True
}

Config.update(config)
```

With this, `Config` gets the values passed to it in `.update()`, and all its values are typed. Neat!

### MultiAgentEnv

An environment is specified in terms of a `coltra.envs.MultiAgentEnv`. It's a rather simple MARL interface
with two unusual class methods - `cls.get_env_creator` and `cls.get_venv`. The first creates a constructor function for the environment and performs
any optional setup that might be necessary. The latter often uses that function and creates a (subprocess) vectorized environment,
with `n` copies of the original environment.

Importantly, **everything** here is multiagent. There is no notion of a single-agent environment - it's just a special case
of a multiagent environment where `num_agents == 1`. This allows us to treat a vectorized environment exactly the same way 
as a regular environment. In a VecEnv, agents have a component in their name describing which of the environments
belong to, e.g. `pursuer_0&env=3`

#### Try it!

A simple interface is using either `MultiGymEnv` for multiagentified Gym environments, or `PettingZooEnv` for PettingZoo envs.

```python
from pettingzoo.sisl import pursuit_v4

from coltra.buffers import Action
from coltra.envs import PettingZooEnv, MultiGymEnv

# env = MultiGymEnv.get_venv(workers=8, env_name="CartPole-v1")  # Creates 8 copies of CartPole

env = PettingZooEnv.get_venv(workers=8, env_fn=pursuit_v4.parallel_env)  # Creates 8 copies of Pursuit, 8 agents each

obs = env.reset()  # Look at the structure of observations

obs, reward, done, info = env.step({agent_id: Action(discrete=env.action_space.sample()) for agent_id in obs})

```

### Observation and Action

This is something that the static-typing/functional-programming nerd in me demanded very loudly. 

Basically, environments always output `Dict[str, Observation]` as observations, and expect `Dict[str, Action]` as actions

`Observation` and `Action` are both defined in `coltra.buffers` and are glorified dataclasses/dictionaries
with some convenience methods. They hold either `np.ndarray`s or `torch.Tensor`s, and perhaps will
be made into explicit generics on that. An Action can hold a continuous action, a discrete action, or a dictionary (not nested) of those.
An Observation can similarly hold a vector or a number of them in a dictionary. There is in principle no difference in how they're
treated, but it allows for multimodal models, e.g. one that receives a vector observation, and raycasts. 
This will (hopefully) make sense when you see how Models are treated.

Note that both Observation and Action can hold either individual values, or batches.

The whole point of this is that now, every environment's output is the same type: `Observation`. This is different
from the usual `gym` model, where the output might be a `np.ndarray` or a `tuple` or a `dict` or who knows what else.
The same is the case for actions.

### Model

Models are the other side of `Observation`, however they don't use `Action` yet. 
A Model inherits from `BaseModel` which in turn inherits from `torch.nn.Module`. It should implement two methods:
`forward(x: Observation, state: Tuple, get_value: bool)` and `value(x: Observation, state: Tuple)`.
Check the detailed return signatures in `coltra.models.BaseModel`, but `forward` should return an action `Distribution`, 
the next recurrent state, and a dictionary with other optional outputs, including value.

**Important note about state** - currently, it's unused and is always an empty tuple. 
It gets carried around to potentially support recurrent policies again, but they're a massive pain, so I'm not sure.
For now just ignore it and always make it an empty tuple.

This is where the `Observation` comes in handy - if you have two types of observations in the environment, e.g. a vector
and an image, you can separately access them with `obs.vector` and `obs.image`.

### Agent

An `Agent` (see: `coltra.agents.Agent`) is the interface between an environment and a model. 
We have two main types of agents: Continuous `CAgent` and discrete `DAgent`. They should be associated with appropriate models.

Conceptually, what the agent does -- it holds a neural network model, accepts an `Observation`, gets an action distribution
from the model, then samples from it in some way, and finally returns the chosen action. 
This happens in the `agent.act` method.

The second important method is `agent.evaluate` which is used during optimization. It takes in a batch of observations and actions,
and returns the respective logprobs, values and entropies, as in all the stuff you need when training. This also properly handles all gradients.

Agents also provide an interface for saving and loading them to the disk.

We also provide several toy agents which take random or constant actions. 
It'd be equally simple to implement an agent performing a specific sequence of actions - I'm sure you see how that can be useful.

Overall this is pretty straight-forward, so check out the implementations for further detail.

One thing that's missing and could be helpful is a mixed continuous-discrete agent for slightly more complex action spaces.
But that's also a relatively rare case since gym doesn't even support that, so it's not here yet.


### MacroAgent

Now we got to the spicy part. Because the environments are multiagent-first, our agents should also be multiagent.
This is handled by the `coltra.groups.MacroAgent` interface. It should conceptually do the same things as an Agent,
except operating on dictionaries of observations/actions.

The simplest case (and the only one implemented at the moment) is `HomogeneousGroup` which is really just 
a thin wrapper around `Agent`, but the interface will make it possible to implement more complex examples.

An important element of a MacroAgent is the `policy_mapping`. In a general MacroAgent, 
you might have several policies which are responsible for different environment agents. 
We do the dispatch based on prefixes. To explain it on an example, a HomogeneousGroup has a simple `policy_mapping`:

```python 
policy_mapping = {"": self.policy_name}
```

Because `""` (empty string) is a prefix of any string, it will match with any agent name, e.g. `pursuer_0`, `evader_1`

Let's say we have two types of agents,`pursuer_x` and `evader_x`, where `x` can be any integer.
We also have two policies, `pursuer` and `evader`. Our policy mapping can then be:

```python
policy_mapping = {"pursuer": "pursuer", "evader": "evader"}
```

Or, if we're being lazy:

```python
policy_mapping = {"p": "pursuer", "e": "evader"}
```

Or even

```python
policy_mapping = {"p": "pursuer", "": "evader"}
```

Each time when we need to match an agent name to a policy name, the group will go through 
all the keys in the policy_mapping, from longest to shortest and see if that key is a prefix of the agent name.
If it is, get that value, otherwise keep searching. If you don't find anything, raise an exception because something's wrong.

This is a relatively new feature, so it still needs to be refined. It assumes that the user Knows What They're Doing (TM), 
so the agents need to be well-named.

RLLib solves the same problem with functions, but functions can't be reliably pickled without some magic. This is simple and works.

#### Try it!

```python
from coltra.models import MLPModel
from coltra.agents import DAgent
from coltra.groups import HomogeneousGroup

from coltra.envs import MultiGymEnv

env = MultiGymEnv.get_venv(8, env_name="CartPole-v1")
model = MLPModel(
    {
        "input_size": env.observation_space.shape[0],
        "num_actions": env.action_space.shape[0],
        "discrete": True
    },
    action_space=env.action_space
)
agent = DAgent(model)
agents = HomogeneousGroup(agent)

# What can you do with model, agent and agents?
```

### Data collection

During any training, we need to collect some data. This is done by the `coltra.collectors.collect_crowd_data` which collects
data with a `HomogeneousGroup` and puts it into a `coltra.buffers.MemoryRecord`. This procedure is pretty simple,
go ahead and check out the code.

The way it works here is that we expect the environment to automatically reset upon completion. 
We collect a fixed number of steps from each of the vectorized envs. and use them for optimization.

### Optimization

This is handled by the `coltra.policy_optimization.CrowdPPOptimizer`. It takes in the data obtained in collection,
and perform gradient updates on the `MacroAgent` with PPO. All the PPO logic is stored in its `train_on_data`, so you don't
need to go through a series of inheritances `PPO -> OnPolicyAlgorithm -> BaseAlgorithm` to know what's going on ;)

### Training

All the components described above are actually everything that you need, see: [At a glance](#at-a-glance). 
But for convenience and proper tensorboard logging, we provide `coltra.trainers.PPOCrowdTrainer` which wraps that logic 
and manages a tensorboard log.

### Scripts

Finally, we have a few scripts that can be used to instantly train or visualize a standard scenario, for example:

```shell
cd scripts
python train_gym.py -c configs/gym_config.yaml -i 500 -e CartPole-v1 -n test_run
python train_pursuit.py -c configs/pursuit_config.yaml -i 500 -n test_run
```

**NOTE:** because nobody other than me ever used this, scripts include logging to my wandb account, which will fail
unless you hack my account. Please don't. You can change it, and in a while I plan to make it managed from the CLI or a file or something.

# Contributing guide

This project is currently *not* encouraging contributions since it's in a volatile state and I need 
to make sure I have a comfortable base that will be somewhat stable and can be built upon.

What I do encourage is feedback -- if something's not clear, or you think could be done better, let me know.
But no promises, since for the moment at least, it's not a community-driven project.

I plan to change this Soon(TM), and if you're reading this, you'll probably be informed about it.

If nevertheless you fell in love with the project and want to help, I have some simple standards:

1. Type hints. Always. Untyped functions scare me.
2. Consistent formatting - just run `black .`
3. Make sure that tests pass. Add new tests when you add new stuff.
4. Keep code clean and readable. Single-variable names are accepted in mathematical parts of the code, nowhere else
