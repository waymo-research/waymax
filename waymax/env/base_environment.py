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

"""Core class definitions for MultiAgentEnvironment.

This environment is designed to work with multiple objects (autonomous driving
vehicle and other objects).
"""
import chex
from dm_env import specs
import jax
from jax import numpy as jnp
from waymax import config as _config
from waymax import datatypes
from waymax import dynamics as _dynamics
from waymax import metrics
from waymax import rewards
from waymax.env import abstract_environment
from waymax.env import typedefs as types


class BaseEnvironment(abstract_environment.AbstractEnvironment):
  """Waymax environment for multi-agent scenarios."""

  def __init__(
      self,
      dynamics_model: _dynamics.DynamicsModel,
      config: _config.EnvironmentConfig,
  ):
    """Constructs a Waymax environment.

    Args:
      dynamics_model: Dynamics model to use which transitions the simulator
        state to the next timestep given an action.
      config: Waymax environment configs.
    """
    self._dynamics_model = dynamics_model
    self._reward_function = rewards.LinearCombinationReward(config.rewards)
    self.config = config

  @property
  def dynamics(self) -> _dynamics.DynamicsModel:
    return self._dynamics_model

  @jax.named_scope('BaseEnvironment.metrics')
  def metrics(self, state: datatypes.SimulatorState) -> types.Metrics:
    """Computes metrics (lower is better) from state."""
    # TODO(b/254483042) Make metric_dict a dataclasses.
    return metrics.run_metrics(
        simulator_state=state, metrics_config=self.config.metrics
    )

  def reset(
      self, state: datatypes.SimulatorState, rng: jax.Array | None = None
  ) -> datatypes.SimulatorState:
    """Initializes the simulation state.

    This initializer sets the initial timestep and fills the initial simulation
    trajectory with invalid values.

    Args:
      state: An uninitialized state of shape (...).
      rng: Optional random number generator for stochastic environments.

    Returns:
      The initialized simulation state of shape (...).
    """
    chex.assert_equal(
        self.config.max_num_objects, state.log_trajectory.num_objects
    )

    # Fills with invalid values (i.e. -1.) and False.
    sim_traj_uninitialized = datatypes.fill_invalid_trajectory(
        state.log_trajectory
    )
    state_uninitialized = state.replace(
        timestep=jnp.array(-1), sim_trajectory=sim_traj_uninitialized
    )
    return datatypes.update_state_by_log(
        state_uninitialized, self.config.init_steps
    )

  def observe(self, state: datatypes.SimulatorState) -> types.Observation:
    """Computes the observation for the given simulation state.

    Here we assume that the default observation is just the simulator state. We
    leave this for the user to override in order to provide a user-specific
    observation function. A user can use this to move some of their model
    specific post-processing into the environment rollout in the actor nodes. If
    they want this post-processing on the accelertor, they can keep this the
    same and implement it on the learner side. We provide some helper functions
    at datatypes.observation.py to help write your own observation functions.

    Args:
      state: Current state of the simulator of shape (...).

    Returns:
      Simulator state as an observation without modifications of shape (...).
    """
    return state

  @jax.named_scope('BaseEnvironment.step')
  def step(
      self,
      state: datatypes.SimulatorState,
      action: datatypes.Action,
      rng: jax.Array | None = None,
  ) -> datatypes.SimulatorState:
    """Advances simulation by one timestep using the dynamics model.

    Args:
      state: The current state of the simulator of shape (...).
      action: The action to apply, of shape (..., num_objects). The
        actions.valid field is used to denote which objects are being controlled
        - objects whose valid is False will fallback to default behavior
        specified by self.dynamics.
      rng: Optional random number generator for stochastic environments.

    Returns:
      The next simulation state after taking an action of shape (...).
    """
    is_controlled = _get_control_mask(state, self.config)
    new_traj = self.dynamics.forward(  # pytype: disable=wrong-arg-types  # jax-ndarray
        action=action,
        trajectory=state.sim_trajectory,
        reference_trajectory=state.log_trajectory,
        is_controlled=is_controlled,
        timestep=state.timestep,
        allow_object_injection=self.config.allow_new_objects_after_warmup,
    )
    return state.replace(sim_trajectory=new_traj, timestep=state.timestep + 1)

  @jax.named_scope('BaseEnvironment.reward')
  def reward(
      self, state: datatypes.SimulatorState, action: datatypes.Action
  ) -> jax.Array:
    """Computes the reward for a transition.

    Args:
      state: The state used to compute the reward at state.timestep of shape
        (...).
      action: The action applied to state of shape (..., num_objects, dim).

    Returns:
      An array of rewards of shape (..., num_objects).
    """
    if self.config.compute_reward:
      agent_mask = datatypes.get_control_mask(
          state.object_metadata, self.config.controlled_object
      )
      return self._reward_function.compute(state, action, agent_mask)
    else:
      reward_spec = _multi_agent_reward_spec(self.config)
      return jnp.zeros(state.shape + reward_spec.shape, dtype=reward_spec.dtype)

  def action_spec(self) -> datatypes.Action:
    # Dynamics model class defines specs for a single agent.
    # Need to expand it to multiple objects.
    single_agent_spec = self.dynamics.action_spec()  # rank 1
    data_spec = specs.BoundedArray(
        shape=(self.config.max_num_objects,) + single_agent_spec.shape,
        dtype=single_agent_spec.dtype,
        minimum=jnp.tile(
            single_agent_spec.minimum[jnp.newaxis, :],
            [self.config.max_num_objects, 1],
        ),
        maximum=jnp.tile(
            single_agent_spec.maximum[jnp.newaxis, :],
            [self.config.max_num_objects, 1],
        ),
    )
    valid_spec = specs.Array(
        shape=(self.config.max_num_objects, 1), dtype=jnp.bool_
    )
    return datatypes.Action(data=data_spec, valid=valid_spec)  # pytype: disable=wrong-arg-types  # jax-ndarray

  def reward_spec(self) -> specs.Array:
    return _multi_agent_reward_spec(self.config)

  def discount_spec(self) -> specs.BoundedArray:
    return specs.BoundedArray(
        shape=tuple(), minimum=0.0, maximum=1.0, dtype=jnp.float32
    )

  def observation_spec(self) -> types.Observation:
    raise NotImplementedError()


def _get_control_mask(
    state: datatypes.SimulatorState, config: _config.EnvironmentConfig
) -> jax.Array:
  """Gets the control mask for a multi-agent environment."""
  if (
      config.controlled_object == _config.ObjectType.VALID
      and not config.allow_new_objects_after_warmup
  ):
    return datatypes.dynamic_index(
        state.sim_trajectory.valid,
        index=config.init_steps - 1,
        axis=-1,
        keepdims=False,
    )
  else:
    return datatypes.get_control_mask(
        state.object_metadata, config.controlled_object
    )


def _multi_agent_reward_spec(
    config: _config.EnvironmentConfig,
) -> specs.Array:
  """Gets the reward spec for a multi-agent environment."""
  return specs.Array(shape=(config.max_num_objects,), dtype=jnp.float32)


# Add MultiAgentEnvironment as an alias for BaseEnvironment, since
# BaseEnvironment already supports executing multiple agents.
MultiAgentEnvironment = BaseEnvironment
