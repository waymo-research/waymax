# Waymax: An accelerated simulator for autonomous driving research.

![Continuous integration](https://github.com/waymo-research/waymax/actions/workflows/ci-build.yml/badge.svg)
[![arXiv](https://img.shields.io/badge/cs.RO-2310.08710-b31b1b?logo=arxiv&logoColor=red)](https://arxiv.org/abs/2310.08710)

[**Documentation**](https://waymo-research.github.io/waymax/docs/)
| [**Tutorials**](https://waymo-research.github.io/waymax/docs/getting_started.html)

## Overview

Waymax is a lightweight, multi-agent, JAX-based simulator for autonomous driving
research based on the [Waymo Open Motion Dataset](https://waymo.com/open/).
Waymax is designed to support research for all aspects of behavior research in
autonomous driving - from closed-loop simulation for planning and sim agent
research to open-loop behavior prediction. Objects (e.g. vehicles, pedestrians)
are represented as bounding boxes, rather than raw sensor outputs, in order to
distill behavior research into its simplest form.

As all components are entirely written in JAX, Waymax is easily distributed and
deployed on hardware accelerators, such as GPUs and
[TPUs](https://cloud.google.com/tpu). Waymax is provided free of charge under
the terms of the [Waymax License Agreement for Non-Commercial Use](https://github.com/waymo-research/waymax/blob/main/LICENSE).


## Installation

Waymax can be installed via pip using the following command:

```
pip install --upgrade pip
pip install git+https://github.com/waymo-research/waymax.git@main#egg=waymo-waymax
```

Please refer to [JAX](https://github.com/jax-ml/jax#installation) for specific
instructions on how to setup JAX with GPU/CUDA support if needed.

### Configure access to Waymo Open Motion Dataset

Waymax is designed to work with the Waymo Open Motion dataset out of the box.

A simple way to configure access via command line is the following:

1.  Apply for [Waymo Open Dataset](https://waymo.com/open) access.

2.  Install [gcloud CLI](https://cloud.google.com/sdk/docs/install)

3.  Run `gcloud auth login <your_email>` with the same email used for step 1.

4.  Run `gcloud auth application-default login`.

If you are using [colab](https://colab.google), run the following inside of the colab after registering in step 1:

```python
from google.colab import auth
auth.authenticate_user()
```

Please reference
[TF Datasets](https://www.tensorflow.org/datasets/gcs#authentication) for
alternative methods to authentication.

## Components

Structurally, Waymax is comprised of a collection of libraries for loading Open
Motion data, visualization, computing common metrics, intelligent sim agents,
and adapters to common RL interfaces such as
[dm-env](https://github.com/deepmind/dm_env). These libraries can be used as
standalone modules, or used together in full closed-loop simulation.

### Dataloading

The `waymax.dataloader` module contains utilities for loading data from the
[Waymo Open Motion Dataset](https://waymo.com/open/).

```python
from waymax import config
from waymax import dataloader

scenarios = dataloader.simulator_state_generator(config.WOD_1_1_0_TRAINING)
scenario = next(scenarios)
```

### Metrics

The `waymax.metrics` module defines commonly used metrics for evaluating agents.
These metrics can be used to evaluate simulated rollouts, or open-loop
predictions from behavior models. Supported metrics include:

-   [Overlap](https://github.com/waymo-research/waymax/tree/main/waymax/metrics/overlap.py)
-   [Offroad](https://github.com/waymo-research/waymax/tree/main/waymax/metrics/roadgraph.py)
-   [Wrong-way](https://github.com/waymo-research/waymax/tree/main/waymax/metrics/roadgraph.py)
-   [Route-following](https://github.com/waymo-research/waymax/tree/main/waymax/metrics/route.py)
-   [Kinematic infeasibility](https://github.com/waymo-research/waymax/tree/main/waymax/metrics/comfort.py)
-   [Log divergence (MSE)](https://github.com/waymo-research/waymax/tree/main/waymax/metrics/imitation.py)

### Agents

The `waymax.agents` module defines intelligent simulated agents for realistic
simulation. Waymax currently supports:

-   Log-playback
-   [IDM](https://github.com/waymo-research/waymax/tree/main/waymax/agents/waypoint_following_agent.py)

### Environments and dynamics

The `waymax.env` module defines a stateless, closed-loop simulator interface as
well as adapters to common RL interfaces such as
[dm-env](https://github.com/deepmind/dm_env) and
[brax](https://github.com/google/brax).

A multi-agent simulation with rewards computed for all agents can be run as
follows:

```python
from waymax import env, config, dynamics, datatypes

# Initialization
dynamics_model = dynamics.InvertibleBicycleModel()
env_config = config.EnvironmentConfig()
scenarios = dataloader.simulator_state_generator(config.WOD_1_1_0_TRAINING)
waymax_env = env.MultiAgentEnvironment(dynamics_model, env_config)

# Rollout
state = waymax_env.reset(next(scenarios))
total_returns = 0
while not state.is_done:
  action = datatypes.Action(data=..., valid=...)  # Compute action here
  total_returns += waymax_env.reward(state, action)
  state = waymax_env.step(state, action)
```

## Tutorials

We provide a few
[colab tutorials](https://github.com/waymo-research/waymax/blob/main/docs/notebooks) for
getting started:

-   [data_demo.ipynb](https://github.com/waymo-research/waymax/blob/main/docs/notebooks/data_demo.ipynb)
    shows how to load the data and use the top-down view visualization.
-   [multi_actors_demo.ipynb](https://github.com/waymo-research/waymax/blob/main/docs/notebooks/multi_actors_demo.ipynb)
    shows how to instantiate multiple agents and run a simple closed-loop
    simulation.
-   [wosac_submission_via_waymax.ipynb](https://github.com/waymo-research/waymax/blob/main/docs/notebooks/wosac_submission_via_waymax.ipynb)
    shows how to create a Waymo Open Sim Agents Challenge submission file.

## Citing Waymax

If you use Waymax for your own research, please cite Waymax in accordance with
the requirements of the Waymax License Agreement for Non-Commercial Use,
including using the following bibtex entry:

```
@inproceedings{waymax,
title={Waymax: An Accelerated, Data-Driven
Simulator for Large-Scale Autonomous Driving Research},
author={Cole Gulino and Justin Fu and Wenjie
Luo and George Tucker and Eli Bronstein and Yiren Lu and Jean Harb and Xinlei Pan and Yan Wang and Xiangyu Chen and John
D. Co-Reyes and Rishabh Agarwal and Rebecca Roelofs and Yao Lu and Nico Montali and Paul Mougin and Zoey Yang and
Brandyn White and Aleksandra Faust, and Rowan McAllister and Dragomir Anguelov and Benjamin Sapp},
booktitle={Proceedings of the Neural Information Processing Systems Track on Datasets and
Benchmarks},year={2023}}
```

## Contact

Please email any questions to [waymax@google.com](mailto:waymax@google.com), or raise an issue on Github.
