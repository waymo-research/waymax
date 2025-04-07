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

"""Library for different object dynamics model for Waymax."""
import abc

import chex
from dm_env import specs
import jax
import jax.numpy as jnp

from waymax import datatypes

CONTROLLABLE_FIELDS = ['x', 'y', 'yaw', 'vel_x', 'vel_y']


class DynamicsModel(abc.ABC):
  """Object dynamics base class."""

  @abc.abstractmethod
  def action_spec(self) -> specs.BoundedArray:
    """Returns the dm_env bounded action spec."""

  @jax.named_scope('DynamicsModel.forward')
  def forward(
      self,
      action: datatypes.Action,
      trajectory: datatypes.Trajectory,
      reference_trajectory: datatypes.Trajectory,
      is_controlled: jax.Array,
      timestep: int,
      allow_object_injection: bool = False,
  ) -> datatypes.Trajectory:
    """Updates a simulated trajectory to the next timestep given an update.

    Args:
      action: Actions to be applied to the trajectory to produce updates at the
        next timestep of shape (..., num_objects).
      trajectory: Simulated trajectory up to the current timestep. This
        trajectory will be updated by this function updated with the trajectory
        update. It is expected that this trajectory will have been updated up to
        `timestep`. This is of shape: (..., num_objects, num_timesteps).
      reference_trajectory: Default trajectory for all objects over the entire
        run segment. Certain fields such as valid are optionally taken from this
        trajectory. This is of shape: (..., num_objects, num_timesteps).
      is_controlled: Boolean array specifying which objects are to be controlled
        by the trajectory update of shape (..., num_objects).
      timestep: Timestep of the current simulation.
      allow_object_injection: Whether to allow new objects to enter the scene.
        If this is set to False, all objects that are not valid at the current
        timestep will not be valid at the next timestep and vice versa.

    Returns:
      Updated trajectory given update from a dynamics model at `timestep` + 1 of
        shape (..., num_objects, num_timesteps).
    """
    chex.assert_equal(trajectory.shape[:-1], action.shape)
    chex.assert_equal_shape_prefix(
        [trajectory, reference_trajectory, is_controlled, action],
        len(is_controlled.shape),
    )
    current_trajectory = datatypes.dynamic_slice(
        inputs=trajectory, start_index=timestep, slice_size=1, axis=-1
    )
    updates = self.compute_update(action, current_trajectory)
    updates.validate()
    return apply_trajectory_update_to_state(
        trajectory_update=updates,
        sim_trajectory=trajectory,
        reference_trajectory=reference_trajectory,
        is_controlled=is_controlled,
        timestep=timestep,
        allow_object_injection=allow_object_injection,
    )

  @abc.abstractmethod
  def compute_update(
      self,
      action: datatypes.Action,
      trajectory: datatypes.Trajectory,
  ) -> datatypes.TrajectoryUpdate:
    """Applies actions to current trajectory to produce next timestep update.

    Only the fields updated as part of the `TrajectoryUpdate` are allowed to be
    updated by the dynamics models. All other fields in `Trajectory` are to be
    updated by the simulator.

    Args:
      action: Actions to be applied to the trajectory to produce updates at the
        next timestep of shape (..., num_objects, dim).
      trajectory: Current trajectory for all agents being controlled, of shape
        (..., num_objects, num_timesteps=1). The actions will be applied to this
        trajectory to produce the update.

    Returns:
      Updated fields of the trajectory after applying the actions for the given
        dynamics model of shape (..., num_objects).
    """

  @abc.abstractmethod
  def inverse(
      self,
      trajectory: datatypes.Trajectory,
      metadata: datatypes.ObjectMetadata,
      timestep: int,
  ) -> datatypes.Action:
    """Computes actions converting traj[timestep] to traj[timestep+1].

    Args:
      trajectory: Full trajectory to compute the inverse actions from of shape
        (..., num_objects, num_timesteps). This trajectory is for the entire
        simulation so that dynamics models can use optimization techniques to
        find the best fitting actions.
      metadata: Metadata describing the types of all objects in the scene, of
        shape (..., num_objects).
      timestep: Current timestep of the simulation.

    Returns:
      Action which will take a set of objects from trajectory[timestep] to
        trajectory[timestep + 1] of shape (..., num_objects, dim).
    """


@jax.named_scope('apply_trajectory_update_to_state')
def apply_trajectory_update_to_state(
    trajectory_update: datatypes.TrajectoryUpdate,
    sim_trajectory: datatypes.Trajectory,
    reference_trajectory: datatypes.Trajectory,
    is_controlled: jax.Array,
    timestep: int,
    allow_object_injection: bool = False,
    use_fallback: bool = False,
) -> datatypes.Trajectory:
  """Applies a TrajectoryUpdate to the sim trajectory at the next timestep.

  When applying a dynamics update, the trajectory will be updated with the
  most recent updates in the trajectory for controlled objects after a dynamics
  update. Fields that are not part of the trajectory update (such as length,
  width, height, valid, etc.) may not be updated in this function.

  For objects not in is_controlled, reference_trajectory is used.
  For objects in is_controlled, but not valid in trajectory_update, fall back to
  constant speed behaviour if the use_fallback flag is on.

  Args:
    trajectory_update: Updated trajectory fields for all objects after the
      dynamics update of shape (..., num_objects, num_timesteps=1).
    sim_trajectory: Simulated trajectory up to the current timestep. This
      trajectory will be modified using the trajectory_update. It is expected
      that this trajectory will have been updated up to `timestep`. This is of
      shape (..., num_objects, num_timesteps).
    reference_trajectory: Default trajectory for all objects over the entire run
      segment. Certain fields such as valid are optionally taken from this
      trajectory. This is of shape: (..., num_objects, num_timesteps).
    is_controlled: Boolean array specifying which objects are to be controlled
      by the trajectory update of shape (..., num_objects).
    timestep: Timestep of the current simulation.
    allow_object_injection: Whether to allow new objects to enter the scene. If
      this is set to False, all objects that are not valid at the current
      timestep will not be valid at the next timestep and visa versa.
    use_fallback: Whether to fall back to constant speed if a controlled agent
      is given an invalid action. Otherwise, the agent will be invalidated.

  Returns:
    Updated trajectory given update from a dynamics model at `timestep` + 1.
  """
  current_traj = datatypes.dynamic_slice(
      inputs=sim_trajectory, start_index=timestep, slice_size=1, axis=-1
  )
  # For is_controlled objects that do not have valid trajectory_update, fall
  # back to trajectory with same velocity.
  fallback_trajectory = current_traj.replace(
      x=current_traj.x + current_traj.vel_x * datatypes.TIME_INTERVAL,
      y=current_traj.y + current_traj.vel_y * datatypes.TIME_INTERVAL,
      yaw=jnp.arctan2(current_traj.vel_y, current_traj.vel_x),
  )
  # For non-controlled objects.
  default_next_traj = datatypes.dynamic_slice(
      inputs=reference_trajectory,
      start_index=timestep + 1,
      slice_size=1,
      axis=-1,
  )

  # Shape: (..., num_objects, 1).
  is_controlled = is_controlled[..., jnp.newaxis]
  chex.assert_equal_shape(
      [is_controlled, trajectory_update.x, default_next_traj.x]
  )
  # Some fields such as the length, width and height of objects are set to be
  # the same for every timestep during data loading and so we don't update these
  # from the current trajectory.
  # TODO(b/266483290): Update z using the (x, y) coordinates of the vehicle.
  replacement_dict = {}
  for field in CONTROLLABLE_FIELDS:
    if use_fallback:
      # Use fallback trajectory if user doesn't not provide valid action.
      new_value = jnp.where(
          trajectory_update.valid,
          trajectory_update[field],
          fallback_trajectory[field],
      )
      # Only update for is_controlled objects from users.
      replacement_dict[field] = jnp.where(
          is_controlled, new_value, default_next_traj[field]
      )
    else:
      new_value = jnp.where(
          is_controlled, trajectory_update[field], default_next_traj[field]
      )
      replacement_dict[field] = new_value

  exist_and_controlled = is_controlled & current_traj.valid
  # For exist_and_controlled objects, valid flags should remain the same as
  # before. For non-exist_and_controlled objects, they can disappear.
  next_valid = jnp.where(
      exist_and_controlled,
      current_traj.valid,
      current_traj.valid & default_next_traj.valid,
  )
  if allow_object_injection:
    # Additionally allow non-exist_and_controlled objects to appear.
    next_valid = jnp.where(
        exist_and_controlled, next_valid, default_next_traj.valid
    )
  replacement_dict['valid'] = next_valid

  # Use timestamp micros from the logged trajectory.
  # This assumes all dynamics have the same control frequency as the logs.
  replacement_dict['timestamp_micros'] = default_next_traj.timestamp_micros
  updated_traj = current_traj.replace(**replacement_dict)

  # Update the simulated trajectory in the simulator state.
  return datatypes.update_by_slice_in_dim(
      inputs=sim_trajectory,
      updates=updated_traj,
      inputs_start_idx=timestep + 1,
      updates_start_idx=0,
      slice_size=1,
      axis=-1,
  )
