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

"""Waymax environment for tasks relating to Planning for the ADV."""

from typing import Sequence

from absl import logging
from dm_env import specs
import jax
import jax.numpy as jnp
from waymax import config as _config
from waymax import datatypes
from waymax import dynamics as _dynamics
from waymax import metrics
from waymax.agents import actor_core
from waymax.env import base_environment as _env
from waymax.env import typedefs as types
from waymax.utils import geometry


class PlanningAgentDynamics(_dynamics.DynamicsModel):
  """A dynamics wrapper for converting multi-agent dynamics to single-agent."""

  def __init__(self, multi_agent_dynamics: _dynamics.DynamicsModel):
    """Initializes with batch prefix dimensions."""
    super().__init__()
    self.wrapped_dynamics = multi_agent_dynamics

  def action_spec(self) -> specs.BoundedArray:
    """Action spec of the action containing the bounds."""
    return self.wrapped_dynamics.action_spec()

  @jax.named_scope('PlanningAgentDynamics.compute_update')
  def compute_update(
      self,
      action: datatypes.Action,
      trajectory: datatypes.Trajectory,
  ) -> datatypes.TrajectoryUpdate:
    """Computes the pose and velocity updates at timestep."""

    # (..., action_dim) --> (..., num_objects, action_dim)
    def tile_for_obj_dimension(x):
      return jnp.repeat(x[..., jnp.newaxis, :], trajectory.num_objects, axis=-2)
    tiled_action = jax.tree_util.tree_map(tile_for_obj_dimension, action)
    tiled_action.validate()
    return self.wrapped_dynamics.compute_update(tiled_action, trajectory)

  @jax.named_scope('PlanningAgentDynamics.forward')
  def forward(
      self,
      action: datatypes.Action,
      trajectory: datatypes.Trajectory,
      log_trajectory: datatypes.Trajectory,
      is_controlled: jax.Array,
      timestep: int,
      allow_new_objects: bool = True,
  ) -> datatypes.Trajectory:
    """Updates a simulated trajectory to the next timestep given an update.

    Runs the forward model for the planning agent by taking in a single object's
    action and tiling it for all others and then running the wrapped action.

    Args:
      action: Actions to be applied to the trajectory to produce updates at the
        next timestep of shape (..., dim).
      trajectory: Simulated trajectory up to the current timestep. This
        trajectory will be updated by this function updated with the trajectory
        update. It is expected that this trajectory will have been updated up to
        `timestep`. This is of shape: (..., num_objects, num_timesteps).
      log_trajectory: Logged trajectory for all objects over the entire run
        segment. Certain fields such as valid are optionally taken from this
        trajectory. This is of shape: (..., num_objects, num_timesteps).
      is_controlled: Boolean array specifying which objects are to be controlled
        by the trajectory update of shape (..., num_objects).
      timestep: Timestep of the current simulation.
      allow_new_objects: Whether to allow new objects to enter the secene. If
        this is set to False, all objects that are not valid at the current
        timestep will not be valid at the next timestep and visa versa.

    Returns:
      Updated trajectory given update from a dynamics model at `timestep` + 1
        of shape (..., num_objects, num_timesteps).
    """
    # (..., action_dim) --> (..., num_objects, action_dim).
    tiled_action_data = jnp.repeat(
        action.data[..., jnp.newaxis, :], trajectory.num_objects, axis=-2
    )
    tiled_valid = jnp.repeat(
        action.valid[..., jnp.newaxis], trajectory.num_objects, axis=-1
    )
    tiled_action = datatypes.Action(data=tiled_action_data, valid=tiled_valid)
    tiled_action.validate()
    return self.wrapped_dynamics.forward(
        tiled_action,
        trajectory,
        log_trajectory,
        is_controlled,
        timestep,
    )

  def inverse(
      self,
      trajectory: datatypes.Trajectory,
      metadata: datatypes.ObjectMetadata,
      timestep: int,
  ) -> datatypes.Action:
    """Computes actions converting traj[timestep] to traj[timestep+1].

    Runs the wrapped dynamics inverse and slices out the sdc's action
    specifically.

    Args:
      trajectory: Full trajectory to compute the inverse actions from of shape
        (..., num_objects, num_timesteps). This trajectory is for the entire
        simulation so that dynamics models can use sophisticated otpimization
        techniques to find the best fitting actions.
      metadata: Metadata on all objects in the scene which contains information
        about what types of objects are in the scene of shape (...,
        num_objects).
      timestep: Current timestpe of the simulation.

    Returns:
      Action which will take a set of objects from trajectory[timestep] to
        trajectory[timestep + 1] of shape (..., num_objects, dim).
    """
    multi_agent_action = self.wrapped_dynamics.inverse(
        trajectory, metadata, timestep
    )
    return datatypes.select_by_onehot(
        multi_agent_action, metadata.is_sdc, keepdims=False
    )


class PlanningAgentEnvironment(_env.BaseEnvironment):
  """An environment wrapper allowing for controlling a single agent.

  The PlanningAgentEnvironment inherits from a multi-agent BaseEnvironment
  to build a single-agent environment by returning only the observations and
  rewards corresponding to the ego-agent (i.e. ADV).

  Note that while the action and reward no longer have an obj dimension as
  expected for a single agent env, the observation retains the obj dimension
  set to 1 to conform with the observation datastructure.
  """

  # TODO: Move to the new sim agent interface when available.
  def __init__(
      self,
      dynamics_model: _dynamics.DynamicsModel,
      config: _config.EnvironmentConfig,
      sim_agent_actors: Sequence[actor_core.WaymaxActorCore] = (),
  ) -> None:
    """Constructs the single agent wrapper.

    Args:
      dynamics_model: Dynamics model that controls how we update the state given
        a planning agent action.
      config: Configuration of the environment.
      sim_agent_actors: Sim agents as Waymax actors used to update other agents
        in the scene besides the ADV. Note the actions generated by the sim
        agents correspond to abstract_dynamics.TrajectoryUpdate.
    """
    self._planning_agent_dynamics = PlanningAgentDynamics(dynamics_model)
    self._state_dynamics = _dynamics.StateDynamics()
    if config.controlled_object != _config.ObjectType.SDC:
      raise ValueError(
          f'controlled_object {config.controlled_object} must be SDC for'
          ' planning agent environment.'
      )
    self._sim_agent_actors = sim_agent_actors
    super().__init__(
        dynamics_model=self._planning_agent_dynamics, config=config
    )

  @jax.named_scope('PlanningAgentEnvironment.metrics')
  def metrics(self, state: datatypes.SimulatorState) -> types.Metrics:
    """Computes the metrics for the single agent wrapper.

    The metrics to be computed are based on those specified by the configuration
    passed into the environment. This runs metrics that may be specific to the
    planning agent case.

    Args:
      state: State of simulation to compute the metrics for. This will compute
        metrics for the timestep corresponding to `state.timestep` of shape
        (...).

    Returns:
      Dictionary from metric name to metrics.MetricResult which represents the
        metrics calculated at `state.timestep`. All metrics assumed to be shaped
        (..., num_objects=1) unless specified in the metrics implementation.
    """
    metric_dict = metrics.run_metrics(state, self.config.metrics)
    # The following metrics need to be selected by one hot. For each, we look
    # if they're in the metric_dict, and if so, we select by onehot and replace
    # the metric in the original metric dictionary.
    multi_agent_metrics_names = ('log_divergence', 'overlap', 'offroad')
    for metric_name in multi_agent_metrics_names:
      if metric_name in metric_dict:
        one_metric_dict = {metric_name: metric_dict[metric_name]}
        one_hot_metric = datatypes.select_by_onehot(
            one_metric_dict, state.object_metadata.is_sdc, keepdims=False
        )
        metric_dict[metric_name] = one_hot_metric[metric_name]

    if self.config.metrics.run_sdc_kinematic_infeasibility:
      # Since initially the first state has a time step of
      # self.config.init_steps - 1, and the transition from
      # self.config.init_steps - 2 to self.config.init_steps - 1 is not
      # necessarily kinematically feasible, so we choose to ignore the first
      # state's sdc_kim value and set it to 0 (kinematically feasible) because
      # the action is not chosen by the actor and is thus not clipped.
      kim_metric_valid = state.timestep > self.config.init_steps - 1
      kim_metric = metric_dict['sdc_kinematic_infeasibility']
      kim_metric = kim_metric.replace(
          value=kim_metric.value * kim_metric_valid,
          valid=kim_metric.valid & kim_metric_valid,
      )
      metric_dict['sdc_kinematic_infeasibility'] = datatypes.select_by_onehot(
          kim_metric, state.object_metadata.is_sdc, keepdims=False
      )
    return metric_dict

  @jax.named_scope('PlanningAgentEnvironment.reward')
  def reward(
      self, state: datatypes.SimulatorState, action: datatypes.Action
  ) -> jax.Array:
    """Computes the reward for a transition.

    Args:
      state: State of simulation to compute the metrics for. This will compute
        reward for the timestep corresponding to `state.timestep` of shape
        (...).
      action: The action applied for the state.

    Returns:
      A float (...) tensor of rewards for the single agent.
    """
    # Shape: (..., num_objects).
    multi_agent_reward = super().reward(state, action)
    # After onehot, shape: (...)
    return datatypes.select_by_onehot(
        multi_agent_reward, state.object_metadata.is_sdc, keepdims=False
    )

  def action_spec(self) -> datatypes.Action:
    # Shape: (..., num_objects, ndim).
    multi_agent_spec = super().action_spec()
    prefix_shape = multi_agent_spec.data.shape[:-2]

    # Remove the object dimension from the action spec.
    data_spec = specs.BoundedArray(
        shape=prefix_shape + multi_agent_spec.data.shape[-1:],
        dtype=multi_agent_spec.data.dtype,
        # Note that `multi_agent_spec` are same for all objects, thus we take
        # the first one's value for min/max.
        minimum=multi_agent_spec.data.minimum[..., 0, :],  # pytype: disable=attribute-error  # jax-ndarray
        maximum=multi_agent_spec.data.maximum[..., 0, :],  # pytype: disable=attribute-error  # jax-ndarray
    )
    # Shape: (..., num_objects, 1).
    valid_spec = specs.Array(
        shape=prefix_shape + (1,),
        dtype=multi_agent_spec.valid.dtype,
    )
    return datatypes.Action(data=data_spec, valid=valid_spec)  # pytype: disable=wrong-arg-types  # jax-ndarray

  @jax.named_scope('PlanningAgentEnvironment.step')
  def step(
      self, state: datatypes.SimulatorState, action: datatypes.Action
  ) -> datatypes.SimulatorState:
    """Advances simulation by one timestep using the dynamics model.

    Args:
      state: The current state of the simulator of shape (...).
      action: The action to apply, of shape (..., num_objects). The
        actions.valid field is used to denote which objects are being controlled
        - objects whose valid is False will fallback to default behavior
        specified by self.dynamics.

    Returns:
      The next simulation state after taking an action of shape (...).
    """
    planning_agent_action = self._planning_agent_dynamics.compute_update(
        action, state.current_sim_trajectory
    ).as_action()
    planning_agent_controlled = state.object_metadata.is_sdc

    merged_action = planning_agent_action
    merged_controlled = planning_agent_controlled
    # Do not control objects which are initialized in a overlap
    # (likely an articulated bus).
    is_controllable = ~_initialized_overlap(state.log_trajectory)

    for agent in self._sim_agent_actors:
      agent_output = agent.select_action({}, state, None, None)  # pytype: disable=wrong-arg-types
      if agent_output.actor_state:
        logging.log_first_n(
            logging.WARNING,
            'Agent output returned actor_state but using actor_state is not'
            ' currently implemented.',
            1,
        )
      action = agent_output.action
      controlled_by_sim = agent_output.is_controlled & is_controllable
      merged_action_data = jnp.where(
          controlled_by_sim[..., jnp.newaxis], action.data, merged_action.data
      )
      merged_action_valid = jnp.where(
          controlled_by_sim[..., jnp.newaxis], action.valid, merged_action.valid
      )
      merged_action = datatypes.Action(
          data=merged_action_data, valid=merged_action_valid
      )
      merged_controlled = merged_controlled | controlled_by_sim

    new_traj = self._state_dynamics.forward(  # pytype: disable=wrong-arg-types  # jax-ndarray
        action=merged_action,
        trajectory=state.sim_trajectory,
        reference_trajectory=state.log_trajectory,
        is_controlled=merged_controlled,
        timestep=state.timestep,
        allow_object_injection=self.config.allow_new_objects_after_warmup,
    )
    return state.replace(sim_trajectory=new_traj, timestep=state.timestep + 1)

  def reward_spec(self) -> specs.Array:
    """Specify the reward spec as just for one object."""
    return specs.Array(shape=(), dtype=jnp.float32)


def _initialized_overlap(log_trajectory: datatypes.Trajectory) -> jax.Array:
  """Return a mask for objects initialized in a overlap state.

  This function returns a boolean mask indicating if an object is in a
  overlap state at timestep 0 in the logged trajectory. This function
  can be used to prune out certain objects that are initialized in an
  overlap, such as articulated buses and pedestrians in a PUDO situation.

  Args:
    log_trajectory: A trajectory of shape (..., num_objects, num_timesteps).

  Returns:
    A [..., objects] boolean tensor of overlap masks.
  """
  trajectory = datatypes.dynamic_index(
      log_trajectory, 0, axis=-1, keepdims=False
  )
  # Shape: (..., num_objects, num_objects).
  traj_5dof = trajectory.stack_fields(['x', 'y', 'length', 'width', 'yaw'])
  pairwise_overlaps = geometry.compute_pairwise_overlaps(traj_5dof)
  # Shape: (..., num_objects).
  return jnp.any(pairwise_overlaps, axis=-1)
