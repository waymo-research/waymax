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

"""Abstract interface for the Waymax environment."""
import abc

from dm_env import specs
import jax
import jax.numpy as jnp
from waymax.env import typedefs as types


class AbstractEnvironment(abc.ABC):
  """A stateless environment interface for Waymax."""

  @abc.abstractmethod
  def reset(
      self, scenario: types.GenericScenario, rng: jax.Array | None = None
  ) -> types.GenericState:
    """Initializes a simulation state.

    This method allows the environment to perform optional postprocessing
    on the state before the episode begins. By default this method is a
    no-op.

    Args:
      scenario: Scenario used to generate the initial state.
      rng: Optional random number generator for stochastic environments.

    Returns:
      The initialized simulation state.
    """

  @abc.abstractmethod
  def step(
      self,
      state: types.GenericState,
      actions: types.GenericAction,
      rng: jax.Array | None = None,
  ) -> types.GenericState:
    """Advances the simulation by one timestep.

    Args:
      state: The current state of the simulator.
      actions: Action to apply to the state to produce the updated simulator
        state.
      rng: Optional random number generator for stochastic environments.

    Returns:
      The next simulation state after taking an action.
    """

  @abc.abstractmethod
  def reward(
      self, state: types.GenericState, action: types.GenericAction
  ) -> jax.Array:
    """Computes the reward for a transition.

    Args:
      state: The state used to compute the reward.
      action: The action applied to state.

    Returns:
      A (..., num_objects) tensor of rewards.
    """

  @abc.abstractmethod
  def metrics(self, state: types.GenericState) -> types.Metrics:
    """Computes a set of metrics which score a given simulator state.

    Args:
      state: The state used to compute the metrics.

    Returns:
      A mapping from metric name to metrics which evaluate a simulator state at
        state.timestep where all of the metrics are of shape (..., num_objects).
    """

  @abc.abstractmethod
  def observe(self, state: types.GenericState) -> types.Observation:
    """Computes the observation of the simulator for the actor.

    Args:
      state: The state used to compute the observation.

    Returns:
      An observation of the simulator state for the given timestep of shape
        (...).
    """

  @abc.abstractmethod
  def action_spec(self) -> types.GenericAction:
    """Returns the action specs of the environment without batch dimension.

    Returns:
      The action specs represented as a PyTree where the leaves
        are instances of specs.Array.
    """

  @abc.abstractmethod
  def reward_spec(self) -> specs.Array:
    """Returns the reward specs of the environment without batch dimension."""

  @abc.abstractmethod
  def discount_spec(self) -> specs.BoundedArray:
    """Returns the discount specs of the environment without batch dimension."""

  @abc.abstractmethod
  def observation_spec(self) -> types.PyTree:
    """Returns the observation specs of the environment without batch dimension.

    Returns:
      The observation specs represented as a PyTree where the
        leaves are instances of specs.Array.
    """

  def termination(self, state: types.GenericState) -> jax.Array:
    """Returns whether the current state is an episode termination.

    A termination marks the end of an episode where the cost-to-go from
    this state is 0.

    The equivalent step type in DMEnv is dm_env.termination.

    Args:
      state: The current simulator state.

    Returns:
      A boolean (...) tensor indicating whether the current state is the end of
        an episode as a termination.
    """
    return jnp.zeros(state.shape, dtype=bool)

  def truncation(self, state: types.GenericState) -> jax.Array:
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
    truncate = state.remaining_timesteps < 1
    return (jnp.ones(state.shape) * truncate).astype(jnp.bool_)
