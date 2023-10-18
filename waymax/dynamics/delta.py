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

"""Library for different delta action space object dynamics models for Waymax.

Delta dynamics modeled can be applied in global coordinates or in the local
(object) coordinate frame.
"""
from dm_env import specs
import jax
import jax.numpy as jnp
import numpy as np

from waymax import datatypes
from waymax.dynamics import abstract_dynamics
from waymax.utils import geometry


class DeltaGlobal(abstract_dynamics.DynamicsModel):
  """Dynamics model operating in global coordinates.

  Deltas (displacements) are computed in the global coordinate frame,
  regardless of the orientation of the objects.
  """

  def __init__(
      self,
      dt: float = 0.1,
      max_dx: float = 6.0,
      max_dy: float = 6.0,
      max_dyaw: float = jnp.pi,
  ):
    """Constructs the DeltaGlobal dynamics model.

    Args:
      dt: Time interval (in seconds) between simulator steps.
      max_dx: Maximum allowed change in x-coordinate over time delta `dt`.
      max_dy: Maximum allowed change in y-coordinate over time delta `dt`.
      max_dyaw: Maximum allowed change in heading (radians) over time delta
        `dt`.
    """
    self._dt = dt
    self._max_dx = max_dx
    self._max_dy = max_dy
    self._max_dyaw = max_dyaw

  def action_spec(self) -> specs.BoundedArray:
    """Action spec for the delta global action space."""
    return specs.BoundedArray(
        # Last dim: (dx, dy, dyaw) in units (meters, meters, radians).
        shape=(3,),
        dtype=np.float32,
        minimum=np.array([-self._max_dx, -self._max_dy, -self._max_dyaw]),
        maximum=np.array([self._max_dx, self._max_dy, self._max_dyaw]),
    )

  @jax.named_scope('DeltaGlobal.compute_update')
  def compute_update(
      self,
      action: datatypes.Action,
      trajectory: datatypes.Trajectory,
  ) -> datatypes.TrajectoryUpdate:
    """Computes the pose and velocity updates at timestep.

    Args:
      action: Actions to take. Has shape (..., num_objects).
      trajectory: Trajectory to be updated. Has shape of (..., num_objects,
        num_timesteps=1).

    Returns:
      The trajectory update for timestep of shape
        (..., num_objects, num_timesteps=1).
    """
    dx, dy, dyaw = jnp.split(action.data, 3, axis=-1)
    vel_x = dx / self._dt
    vel_y = dy / self._dt
    return datatypes.TrajectoryUpdate(
        x=trajectory.x + dx,
        y=trajectory.y + dy,
        yaw=trajectory.yaw + dyaw,
        vel_x=vel_x,
        vel_y=vel_y,
        valid=trajectory.valid & action.valid,
    )

  def _clip_values(self, action: jax.Array) -> jax.Array:
    """Clips action values to the action spec.

    Args:
      action: A tensor of shape (..., 3) containing x, y, and yaw values.

    Returns:
      A tensor of shape (..., 3) containing action values clipped to the
      minimum/maximum bounds.
    """
    x = jnp.clip(
        action[..., 0],
        self.action_spec().minimum[0],
        self.action_spec().maximum[0],
    )
    y = jnp.clip(
        action[..., 1],
        self.action_spec().minimum[1],
        self.action_spec().maximum[1],
    )
    yaw = geometry.wrap_yaws(action[..., 2])
    return jnp.stack([x, y, yaw], axis=-1)

  def inverse(
      self,
      trajectory: datatypes.Trajectory,
      metadata: datatypes.ObjectMetadata,
      timestep: int,
  ) -> datatypes.Action:
    """Runs inverse dynamics model to infer actions for specified timestep.

    Args:
      trajectory: A Trajectory used to infer actions of shape (..., num_objects,
        num_timesteps).
      metadata: Object metadata for the trajectory of shape (..., num_objects).
      timestep: Index of time for actions.

    Returns:
      An Action that converts traj[timestep] to traj[timestep+1] of shape
        (..., num_objects, dim=2).
    """
    # Form a tensor of shape (..., num_objects, num_timesteps, 3)
    xy_yaw_ = jnp.stack([trajectory.x, trajectory.y, trajectory.yaw], axis=-1)
    # Sliced before/after values of shape (..., num_objects, num_timesteps=2, 3)
    xy_yaw = jax.lax.dynamic_slice_in_dim(
        xy_yaw_, start_index=timestep, slice_size=2, axis=-2
    )
    raw_action_array = xy_yaw[..., 1, :] - xy_yaw[..., 0, :]

    valids = jax.lax.dynamic_slice_in_dim(
        trajectory.valid, start_index=timestep, slice_size=2, axis=-1
    )
    valid = valids[..., 0:1] & valids[..., 1:2]

    # Defaults to 0s for invalid actions.
    action_array = jnp.where(valid, raw_action_array, 0.0)
    action_array = self._clip_values(action_array)

    return datatypes.Action(data=action_array, valid=valid)


class DeltaLocal(DeltaGlobal):
  """Dynamics model operating in global coordinates.

  Actions are defined as (dx, dy, dyaw) in local coordinates.
  """

  @jax.named_scope('DeltaLocal.compute_update')
  def compute_update(
      self,
      action: datatypes.Action,
      trajectory: datatypes.Trajectory,
  ) -> datatypes.TrajectoryUpdate:
    """Converts to global actions and calls DeltaGlobal.compute_update.

    Args:
      action: Actions to take. Has shape (..., num_objects).
      trajectory: Trajectory to be updated. Has shape of (..., num_objects,
        num_timesteps=1).

    Returns:
      The trajectory update for timestep of shape
        (..., num_objects, num_timesteps=1).
    """
    transf_yaw = trajectory.yaw[..., 0]
    transf_valid = trajectory.valid[..., 0:1]

    rotation_mat = geometry.rotation_matrix_2d(transf_yaw)
    # (..., 2, 2) x (..., 2) --> (..., 2).
    rotated_xy = jnp.matmul(
        rotation_mat, action.data[..., :2, jnp.newaxis], precision='float32'
    )[..., 0]

    # Concatenate last dimension for dx, dy and dyaw. Shape (..., 3)
    action_array = jnp.concatenate([rotated_xy, action.data[..., 2:3]], axis=-1)
    global_actions = datatypes.Action(
        data=action_array, valid=transf_valid & action.valid
    )
    return super().compute_update(global_actions, trajectory)

  def inverse(
      self,
      trajectory: datatypes.Trajectory,
      metadata: datatypes.ObjectMetadata,
      timestep: int,
  ) -> datatypes.Action:
    """Calls DeltaGlobal's inverse and converts to local coordinates.

    Args:
      trajectory: A Trajectory used to infer actions of shape (..., num_objects,
        num_timesteps=1).
      metadata: Object metadata for the trajectory of shape (..., num_objects).
      timestep: index of time for actions.

    Returns:
      An Action that converts traj[timestep] to traj[timestep+1] of shape
        (..., num_objects, dim=2).
    """
    global_actions = super().inverse(trajectory, metadata, timestep)

    # Yaw in global coordinates.
    current_yaw = jax.lax.dynamic_slice_in_dim(
        trajectory.yaw, timestep, slice_size=1, axis=-1
    )[..., 0]
    current_valid = jax.lax.dynamic_slice_in_dim(
        trajectory.valid, timestep, slice_size=1, axis=-1
    )[..., 0:1]

    rotation_global_to_local = geometry.rotation_matrix_2d(-current_yaw)
    # (..., 2, 2) x (..., 2) --> (..., 2).
    local_xy = jnp.matmul(
        rotation_global_to_local,
        global_actions.data[..., :2, jnp.newaxis],
        precision='float32',
    )[..., 0]

    # Concatenate last dimension for dx, dy and dyaw. Shape (..., 3)
    local_action_array = jnp.concatenate(
        [local_xy, global_actions.data[..., 2:3]], axis=-1
    )
    local_action_array = self._clip_values(local_action_array)

    return datatypes.Action(
        data=local_action_array, valid=current_valid & global_actions.valid
    )
