# Copyright 2023 The Waymax Authors.
#
# Licensed under the Waymax License Agreement for Non-commercial Use
# Use (the "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at
#
#     https://github.com/waymo-research/waymax/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Library for wrapping Waymax environments in a Brax-like interface.

For more information on the Brax interface see: https://github.com/google/brax.

The Waymax/Brax interface primarily differs from the Google/Brax interface in
the reset function. Because Waymax uses data to instantiate a new episode,
the reset function requires a SimulatorState argument, whereas the Google/Brax
interface requires only a random key.
"""
from typing import Any

import chex
from dm_env import specs
from flax import struct
import jax
from jax import numpy as jnp
from waymax import config as _config
from waymax import datatypes
from waymax import dynamics
from waymax.env import abstract_environment
from waymax.env import typedefs as types


@chex.dataclass(frozen=True)
class TimeStep:
  """Container class for Waymax transitions.

  Attributes:
    state: The current simulation state of shape (...).
    observation: The current observation of shape (..,).
    reward: The reward obtained in the current transition of shape (...,
      num_objects).
    done: A boolean array denoting the end of an episode of shape (...).
    discount: An array of discount values of shape (...).
    metrics: Optional dictionary of metrics.
    info: Optional dictionary of arbitrary logging information.
  """

  state: datatypes.SimulatorState
  observation: types.Observation
  reward: jax.Array
  done: jax.Array
  discount: jax.Array
  metrics: types.Metrics = struct.field(default_factory=dict)
  info: dict[str, Any] = struct.field(default_factory=dict)

  @property
  def shape(self) -> tuple[int, ...]:
    """Shape of TimeStep."""
    return self.state.shape

  def __eq__(self, other: Any) -> bool:
    return datatypes.compare_all_leaf_nodes(self, other)


class BraxWrapper:
  """Brax-like interface wrapper for the Waymax environment."""

  def __init__(
      self,
      wrapped_env: abstract_environment.AbstractEnvironment,
      dynamics_model: dynamics.DynamicsModel,
      config: _config.EnvironmentConfig,
  ) -> None:
    """Constracts the Brax wrapper over a Waymax environment.

    Args:
      wrapped_env: Waymax environment to wrap with the Brax interface.
      dynamics_model: Dynamics model to use which transitions the simulator
        state to the next timestep given an action.
      config: Waymax environment configs.
    """
    self._wrapped_env = wrapped_env
    self.dynamics = dynamics_model
    self.config = config

  def metrics(self, state: datatypes.SimulatorState) -> types.Metrics:
    """Computes metrics (lower is better) from state."""
    return self._wrapped_env.metrics(state)

  def reset(self, state: datatypes.SimulatorState) -> TimeStep:
    """Resets the environment and initializes the simulation state.

    This initializer sets the initial timestep and fills the initial simulation
    trajectory with invalid values.

    Args:
      state: An uninitialized state.

    Returns:
      The initialized simulation state.
    """
    initial_state = self._wrapped_env.reset(state)
    return TimeStep(
        state=initial_state,
        observation=self.observe(initial_state),
        done=self.termination(initial_state),
        reward=jnp.zeros(state.shape + self.reward_spec().shape),
        discount=jnp.ones(state.shape + self.discount_spec().shape),
        metrics=self.metrics(initial_state),
    )

  def observe(self, state: datatypes.SimulatorState) -> types.Observation:
    """Computes the observation for the given simulation state."""
    return self._wrapped_env.observe(state)

  def step(self, timestep: TimeStep, action: datatypes.Action) -> TimeStep:
    """Advances simulation by one timestep using the dynamics model.

    Args:
      timestep: The timestep containing the current state.
      action: The action to apply, of shape (..., num_objects). The
        actions.valid field is used to denote which objects are being controlled
        - objects whose valid is False will fallback to default behavior
        specified by self.dynamics.

    Returns:
      The timestep corresponding to the transition taken.
    """
    next_state = self._wrapped_env.step(timestep.state, action)
    obs = self.observe(next_state)
    reward = self.reward(timestep.state, action)
    termination = self.termination(next_state)
    truncation = self.truncation(next_state)
    done = jnp.logical_or(termination, truncation)
    discount = jnp.logical_not(termination).astype(jnp.float32)
    metric_dict = self.metrics(timestep.state)
    return TimeStep(
        state=next_state,
        reward=reward,
        observation=obs,
        done=done,
        discount=discount,
        metrics=metric_dict,
    )

  def reward(
      self, state: datatypes.SimulatorState, action: datatypes.Action
  ) -> jax.Array:
    """Computes the reward for a transition.

    Args:
      state: The state used to compute the reward at state.timestep.
      action: The action applied to state.

    Returns:
      A (..., num_objects) tensor of rewards.
    """
    return self._wrapped_env.reward(state, action)

  def termination(self, state: datatypes.SimulatorState) -> jax.Array:
    """Returns whether the current state is an episode termination.

    A termination marks the end of an episode where the cost-to-go from
    this state is 0.

    The equivalent step type in DMEnv is dm_env.termination.

    Args:
      state: The current simulator state.

    Returns:
      A boolean (...) tensor indicating whether the current state is the end
        of an episode as a termination.
    """
    return self._wrapped_env.termination(state)

  def truncation(self, state: datatypes.SimulatorState) -> jax.Array:
    """Returns whether the current state should truncate the episode.

    A truncation denotes that an episode has ended due to reaching the step
    limit of an episode. In these cases dynamic programming methods (e.g.
    Q-learning) should still compute cost-to-go assuming the episode will
    continue running.

    The equivalent step type in DMEnv is dm_env.truncation.

    Args:
      state: The current simulator state.

    Returns:
      A boolean (...) tensor indicating whether the current state is the end of
        an episode as a truncation.
    """
    return self._wrapped_env.truncation(state)

  def action_spec(self) -> datatypes.Action:
    """Action spec of the environment."""
    return self._wrapped_env.action_spec()

  def reward_spec(self) -> specs.Array:
    """Reward spec of the environment."""
    return self._wrapped_env.reward_spec()

  def discount_spec(self) -> specs.BoundedArray:
    """Discount spec of the environment."""
    return self._wrapped_env.discount_spec()

  def observation_spec(self) -> types.PyTree:
    """Observation spec of the environment."""
    return self._wrapped_env.observation_spec()
