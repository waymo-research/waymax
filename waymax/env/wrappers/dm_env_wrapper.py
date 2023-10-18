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

"""Library for wrapping Waymax environments in a DMEnv interface."""
from typing import Iterator

import dm_env
from dm_env import specs
import jax
from jax import numpy as jnp
from waymax import config as _config
from waymax import dataloader
from waymax import datatypes
from waymax import dynamics
from waymax.agents import agent_builder
from waymax.env import abstract_environment
from waymax.env import errors
from waymax.env import planning_agent_environment
from waymax.env import typedefs as types


class DMEnvWrapper(dm_env.Environment):
  """A stateful environment wrapper implementing the DMEnv interface."""

  def __init__(
      self,
      data_generator: Iterator[datatypes.SimulatorState],
      stateless_env: abstract_environment.AbstractEnvironment,
      squeeze_scalar_actions: bool = True,
  ) -> None:
    """Initializes the internal state of the Waymax dm_env wrapper.

    Args:
      data_generator: Dataset generator that when called as
        next(run_segments_generator) produces a SimulatorState initialized from
        a new run segment.
      stateless_env: The wrapped stateless environment which in general is a
        base Waymax environment and conforms to its interface.
      squeeze_scalar_actions: Whether to squeeze scalar actions when using a
        discrete action space. This is required for certain ACME interfaces. For
        discrete actions, Acme wants scalar actions, but Waymax always has an
        action dimension.
    """
    self._stateless_env = stateless_env
    self._data_generator = data_generator
    self._sample_state = next(self._data_generator)
    self._cur_state = None
    self._initialized = False
    self._done = False
    self._jitted_stateless_env_step = jax.jit(self._stateless_env.step)
    self._jitted_stateless_env_reward = jax.jit(self._stateless_env.reward)
    self._squeeze_scalar_actions = squeeze_scalar_actions

  @property
  def config(self) -> _config.EnvironmentConfig:
    if hasattr(self._stateless_env, 'config'):
      return self._stateless_env.config
    raise ValueError('The wrapped_env does not have a config.')

  @property
  def simulation_state(self) -> datatypes.SimulatorState:
    """The current simulation state."""
    if not self._initialized:
      raise errors.SimulationNotInitializedError()
    return self._cur_state

  @property
  def stateless_env(self) -> abstract_environment.AbstractEnvironment:
    """The underlying stateless Waymax environment."""
    return self._stateless_env

  def observe(self, state: datatypes.SimulatorState) -> types.Observation:
    """Runs the stateless environment observation function."""
    return self._stateless_env.observe(state)

  def reset(self) -> dm_env.TimeStep:
    """Resets the environment and returns the initial TimeStep."""
    self._cur_state = self.stateless_env.reset(next(self._data_generator))
    self._initialized = True
    if self._cur_state.is_done:
      raise errors.SimulationNotInitializedError()
    self._done = False
    return dm_env.restart(observation=self.observe(self._cur_state))

  def step(self, action: jax.Array) -> dm_env.TimeStep:
    """Advances the state given an action.

    Args:
      action: An action with shape compatible with self.action_spec()

    Returns:
      The TimeStep corresponding to the transition taken by applying
        action to the current state.

    Raises:
      SimulationNotInitializedError: If reset() has not been called before
        this method is called.
      EpisodeAlreadyFinishedError: If this method is called after an episode
        has been terminated or truncated.
    """
    if not self._initialized:
      raise errors.SimulationNotInitializedError()
    if self._done:
      raise errors.EpisodeAlreadyFinishedError()

    # Calculate the reward on the simulation step before updating the state with
    # the given action.
    reward = self._jitted_stateless_env_reward(self._cur_state, action)

    action = jnp.asarray(action)
    # Reshape the action to what the underlying environment wants.
    action = jnp.reshape(
        action,
        self._cur_state.shape + self.stateless_env.action_spec().data.shape,
    )

    action = datatypes.Action(
        data=action, valid=jnp.ones_like(action[..., 0:1], dtype=jnp.bool_)
    )
    action.validate()
    self._cur_state = self._jitted_stateless_env_step(self._cur_state, action)

    termination = self._stateless_env.termination(self._cur_state)
    truncation = self._stateless_env.truncation(self._cur_state)
    done = jnp.logical_or(termination, truncation)
    discount = jnp.logical_not(termination).astype(jnp.float32)
    if jnp.all(done):
      self._done = True
      return dm_env.TimeStep(
          dm_env.StepType.LAST,
          discount=discount,
          reward=reward,
          observation=self.observe(self._cur_state),
      )
    elif jnp.any(done):
      raise NotImplementedError(
          'Batches with mixed step types are not currently supported.'
      )
    return dm_env.transition(
        reward=reward, observation=self.observe(self._cur_state)
    )

  def action_spec(self) -> specs.BoundedArray:
    """The action specs of this environment, without batch dimension."""
    action_spec = self.stateless_env.action_spec().data
    if self._squeeze_scalar_actions and tuple(action_spec.shape) == (1,):
      # Squeeze spec in this special case.
      return specs.BoundedArray(
          shape=(),
          dtype=action_spec.dtype,
          minimum=action_spec.minimum[0],  # pytype: disable=attribute-error  # jax-ndarray
          maximum=action_spec.maximum[0],  # pytype: disable=attribute-error  # jax-ndarray
      )
    else:
      return action_spec  # pytype: disable=bad-return-type  # jax-ndarray

  def discount_spec(self) -> specs.BoundedArray:
    """The discount specs of this environment, without batch dimension."""
    return self.stateless_env.discount_spec()

  def observation_spec(self) -> types.PyTree:
    """The observation specs of this environment, without batch dimension."""
    sample_obs = self.observe(self.stateless_env.reset(self._sample_state))
    batch_ndim = len(sample_obs.batch_dims)

    def array_to_spec(array: jax.Array) -> specs.Array:
      if array.dtype in [jnp.float32, jnp.int32, jnp.bool_]:
        return specs.Array(shape=array.shape[batch_ndim:], dtype=array.dtype)
      else:
        raise NotImplementedError(array.dtype)

    return jax.tree_util.tree_map(array_to_spec, sample_obs)

  def reward_spec(self) -> specs.Array:
    """The reward specs of this environment, without batch dimension."""
    return self.stateless_env.reward_spec()


def make_sdc_dm_environment(
    dynamics_model: dynamics.DynamicsModel,
    data_config: _config.DatasetConfig,
    env_config: _config.EnvironmentConfig,
) -> DMEnvWrapper:
  """Makes a DM environment for controlling SDC only.

  Args:
    dynamics_model: A dynamics model used to transit state of the environment.
    data_config: Config for dataset, see details in config.DatasetConfig
    env_config: Config for environment, see details in config.EnvironmentConfig.

  Returns:
    The single agent (SDC) Waymax DM environment that has not been reset.
  """
  if env_config.controlled_object != _config.ObjectType.SDC:
    raise ValueError(
        f'controlled_object should be SDC, got {env_config.controlled_object}'
    )
  if env_config.sim_agents is not None:
    sim_agent_actors = [
        agent_builder.create_sim_agents_from_config(agent)
        for agent in env_config.sim_agents
    ]
  else:
    sim_agent_actors = []
  dataset_iterator = dataloader.simulator_state_generator(config=data_config)
  single_env = planning_agent_environment.PlanningAgentEnvironment(
      dynamics_model=dynamics_model,
      config=env_config,
      sim_agent_actors=sim_agent_actors,
  )
  return DMEnvWrapper(dataset_iterator, single_env)
