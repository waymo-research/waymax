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

"""Implementation of the bicycle (acceleration, steering) dynamics model.

The bicycle model uses acceleration and curvature as the action space, with
the max/min bounds on acceleration and curvature tuned to minimize error
on the Waymo Open Motion dataset. Unlike the Ackermann steering model,
this dynamics model has an analytical inverse that can be used to compute
expert actions from logged behavior.


This action space always uses the [-1.0, 1.0] as the range for acceleration
and steering commands to be consistent with other RL training pipeline since
many algorithms' hyperparameters are tuned based on this assumption. The actual
acceleration and steering command range can still be specified by `max_accel`
and `max_steering` in the class definition function.
"""

from dm_env import specs
import jax
import jax.numpy as jnp
import numpy as np

from waymax import datatypes
from waymax.dynamics import abstract_dynamics
from waymax.utils import geometry

DynamicsModel = abstract_dynamics.DynamicsModel
# TODO(b/256862507) Determine whether 0.6 is appropriate speed limit.
# This speed limit helps to filter out false positive very large steering value.
_SPEED_LIMIT = 0.6  # Units: m/s


@jax.named_scope('bicycle_model.compute_inverse')
def compute_inverse(
    traj: datatypes.Trajectory,
    timestep: jax.typing.ArrayLike,
    dt: float = 0.1,
    estimate_yaw_with_velocity: bool = True,
) -> datatypes.Action:
  """Runs inverse dynamics model to infer actions for specified timestep.

  Inverse dynamics:
    accel = (new_vel - vel) / dt
    steering = (new_yaw - yaw) / (speed * dt + 1/2 * accel * dt ** 2)

  Args:
    traj: A Trajectory used to infer actions of shape (..., num_objects,
      num_timesteps).
    timestep: Index of time for actions.
    dt: The time step length used in the simulator.
    estimate_yaw_with_velocity: Whether to use the yaw recorded in `traj` for
      estimating the inverse action or use the yaw estimated from velocities. It
      is recommended to set this to True, as using the estimated yaw is
      generally less noisy than using the yaw directly recorded in the
      trajectory.

  Returns:
    An Action that converts traj[timestep] to traj[timestep+1] of shape
      (..., num_objects, dim=2).
  """
  xy_yaw_vel = jnp.stack(
      [traj.x, traj.y, traj.yaw, traj.vel_x, traj.vel_y], axis=-1
  )
  xy_yaw_vel_slice = jax.lax.dynamic_slice_in_dim(
      xy_yaw_vel, start_index=timestep, slice_size=2, axis=-2
  )
  # Each has shape (..., num_timesteps = 2, 1).
  _, _, yaw, vel_x, vel_y = jnp.split(xy_yaw_vel_slice, 5, axis=-1)
  valids = jax.lax.dynamic_slice_in_dim(
      traj.valid, start_index=timestep, slice_size=2, axis=-1
  )
  valid = valids[..., 0:1] & valids[..., 1:2]
  # Calculate acceleration.
  speed = jnp.sqrt(vel_x[..., 0:2, :] ** 2 + vel_y[..., 0:2, :] ** 2)
  new_speed = speed[..., 1:2, :]
  accel = (new_speed - speed[..., 0:1, :]) / dt

  # Calculate steering curvature.
  new_yaw = geometry.wrap_yaws(yaw[..., 1:2, :])
  yaw = geometry.wrap_yaws(yaw[..., 0:1, :])
  if estimate_yaw_with_velocity:
    real_new_yaw = jnp.arctan2(vel_y[..., 1:2, :], vel_x[..., 1:2, :])
  else:
    real_new_yaw = new_yaw
  real_new_yaw = jnp.where(
      jnp.abs(new_speed) <= _SPEED_LIMIT, new_yaw, real_new_yaw
  )
  delta_yaw = geometry.wrap_yaws(real_new_yaw - yaw)
  steering = delta_yaw / (speed[..., 0:1, :] * dt + 0.5 * accel * dt**2)
  # Set steering to 0.0 if speed is 0 to avoid NaN error.
  # When speed is small, delta_yaw sometimes can also be small, so the
  # calculation of steering is affected by the data noise and can lead to
  # overestimation of steering, filtering small speed can help to prevent
  # overestimation of steering.
  steering = jnp.where(jnp.abs(speed[..., 0:1, :]) < _SPEED_LIMIT, 0, steering)
  steering = jnp.where(jnp.abs(speed[..., 1:2, :]) < _SPEED_LIMIT, 0, steering)
  raw_action_array = jnp.concatenate([accel, steering], axis=-1).squeeze(-2)
  action_array = jnp.where(valid, raw_action_array, 0.0)
  return datatypes.Action(data=action_array, valid=valid)


class InvertibleBicycleModel(DynamicsModel):
  """Dynamics model using acceleration and steering curvature for control."""

  def __init__(
      self,
      dt: float = 0.1,
      max_accel: float = 6.0,
      max_steering: float = 0.3,
      normalize_actions: bool = False,
  ):
    """Initializes the bounds of the action space.

    Args:
      dt: The time length per step used in the simulator in seconds.
      max_accel: The maximum acceleration magnitude.
      max_steering: The maximum steering curvature magnitude, which is the
        inverse of the turning radius (the minimum radius of available space
        required for that vehicle to make a circular turn).
      normalize_actions: Whether to normalize the action range to [-1,1] or not.
        By default it uses the unnormalized range and in order to train with RL,
        such as with ACME. Ideally we should normalize the ranges.
    """
    super().__init__()
    self._dt = dt
    self._max_accel = max_accel
    self._max_steering = max_steering
    self._normalize_actions = normalize_actions

  def action_spec(self) -> specs.BoundedArray:
    """Action spec for the acceleration steering continuous action space."""
    if not self._normalize_actions:
      return specs.BoundedArray(
          # last dim: (acceleration, steering)
          shape=(2,),
          dtype=np.float32,
          minimum=np.array([-self._max_accel, -self._max_steering]),
          maximum=np.array([self._max_accel, self._max_steering]),
      )
    else:
      return specs.BoundedArray(
          # last dim: (acceleration, steering)
          shape=(2,),
          dtype=np.float32,
          minimum=np.array([-1.0, -1.0]),
          maximum=np.array([1.0, 1.0]),
      )

  def _clip_values(self, action_array: jax.Array) -> jax.Array:
    """Clip action values to be within the allowable ranges."""
    acc = jnp.clip(
        action_array[..., 0],
        self.action_spec().minimum[0],
        self.action_spec().maximum[0],
    )
    steering = jnp.clip(
        action_array[..., 1],
        self.action_spec().minimum[1],
        self.action_spec().maximum[1],
    )
    return jnp.stack([acc, steering], axis=-1)

  @jax.named_scope('InvertibleBicycleModel.compute_update')
  def compute_update(
      self,
      action: datatypes.Action,
      trajectory: datatypes.Trajectory,
  ) -> datatypes.TrajectoryUpdate:
    """Computes the pose and velocity updates at timestep.

    Forward dynamics:
      new_x = x + vel_x * t + 1/2 * accel * cos(yaw) * t ** 2
      new_y = y + vel_y * t + 1/2 * accel * sin(yaw) * t ** 2
      new_yaw = yaw + steering * (speed * t + 1/2 * accel * t ** 2)
      new_vel = vel + accel * t

    Args:
      action: Actions of shape (..., num_objects) containing acceleration and
        steering controls.
      trajectory: Trajectory to be updated. Has shape of (..., num_objects,
        num_timesteps=1).

    Returns:
      The trajectory update for timestep of shape
        (..., num_objects, num_timesteps=1).
    """
    x = trajectory.x
    y = trajectory.y
    vel_x = trajectory.vel_x
    vel_y = trajectory.vel_y
    yaw = trajectory.yaw
    speed = jnp.sqrt(trajectory.vel_x**2 + trajectory.vel_y**2)

    # Shape: (..., num_objects, 2)
    action_array = self._clip_values(action.data)
    accel, steering = jnp.split(action_array, 2, axis=-1)
    if self._normalize_actions:
      accel = accel * self._max_accel
      steering = steering * self._max_steering
    t = self._dt

    new_x = x + vel_x * t + 0.5 * accel * jnp.cos(yaw) * t**2
    new_y = y + vel_y * t + 0.5 * accel * jnp.sin(yaw) * t**2
    delta_yaw = steering * (speed * t + 0.5 * accel * t**2)
    new_yaw = geometry.wrap_yaws(yaw + delta_yaw)
    new_vel = speed + accel * t
    new_vel_x = new_vel * jnp.cos(new_yaw)
    new_vel_y = new_vel * jnp.sin(new_yaw)
    return datatypes.TrajectoryUpdate(
        x=new_x,
        y=new_y,
        yaw=new_yaw,
        vel_x=new_vel_x,
        vel_y=new_vel_y,
        valid=trajectory.valid & action.valid,
    )

  def inverse(
      self,
      trajectory: datatypes.Trajectory,
      metadata: datatypes.ObjectMetadata,
      timestep: int,
  ) -> datatypes.Action:
    """Runs inverse dynamics model to infer actions for specified timestep.

    Inverse dynamics:
      accel = (new_vel - vel) / dt
      steering = (new_yaw - yaw) / (speed * dt + 1/2 * accel * dt ** 2)

    Args:
      trajectory: A Trajectory used to infer actions (..., num_objects,
        num_timesteps),
      metadata: Object metadata for the trajectory of shape (..., num_objects).
      timestep: Index of time for actions.

    Returns:
      An Action that converts traj[timestep] to traj[timestep+1] of shape
        (..., num_objects, dim=2).
    """
    actions = compute_inverse(trajectory, timestep, self._dt)
    if self._normalize_actions:
      action_array = actions.data
      # accel/steering shape: (..., num_objects)
      accel = action_array[..., 0] / self._max_accel
      steering = action_array[..., 1] / self._max_steering
      # action_array shape: (..., num_objects, 2)
      action_array = jnp.stack([accel, steering], axis=-1)
      action_array = self._clip_values(action_array)
    else:
      action_array = self._clip_values(actions.data)
    return datatypes.Action(data=action_array, valid=actions.valid)
