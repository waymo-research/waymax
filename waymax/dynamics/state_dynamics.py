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

"""Dynamics model for setting state in global coordinates."""
from dm_env import specs
import jax
import numpy as np

from waymax import datatypes
from waymax.dynamics import abstract_dynamics


class StateDynamics(abstract_dynamics.DynamicsModel):
  """Dynamics model for setting state in global coordinates."""

  def __init__(self):
    """Initializes the StateDynamics."""

  def action_spec(self) -> specs.BoundedArray:
    """Action spec for the delta global action space."""
    return specs.BoundedArray(
        shape=(len(abstract_dynamics.CONTROLLABLE_FIELDS),),
        dtype=np.float32,
        minimum=-float('inf'),
        maximum=float('inf'),
    )

  def compute_update(
      self,
      action: datatypes.Action,
      trajectory: datatypes.Trajectory,
  ) -> datatypes.TrajectoryUpdate:
    """Computes the pose and velocity updates at timestep.

    This dynamics will directly set the next x, y, yaw, vel_x, and vel_y based
    on the action.

    Args:
      action: Actions to take. Has shape (..., num_objects).
      trajectory: Trajectory to be updated. Has shape of (..., num_objects,
        num_timesteps=1).

    Returns:
      The trajectory update for timestep.
    """
    del trajectory  # Not used.
    return datatypes.TrajectoryUpdate(
        x=action.data[..., 0:1],
        y=action.data[..., 1:2],
        yaw=action.data[..., 2:3],
        vel_x=action.data[..., 3:4],
        vel_y=action.data[..., 4:5],
        valid=action.valid,
    )

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
      metadata: Object metadata for the trajectory.
      timestep: Index of time for actions.

    Returns:
      An Action that converts traj[timestep] to traj[timestep+1].
    """
    del metadata  # Not used.
    # Shape: (..., num_objects, num_timesteps, 5)
    stacked = trajectory.stack_fields(abstract_dynamics.CONTROLLABLE_FIELDS)
    # Shape: (..., num_objects, num_timesteps=1, 5)
    stacked = jax.lax.dynamic_slice_in_dim(
        stacked, start_index=timestep + 1, slice_size=1, axis=-2
    )
    valids = jax.lax.dynamic_slice_in_dim(
        trajectory.valid, start_index=timestep + 1, slice_size=1, axis=-1
    )
    # Slice out timestep dimension.
    return datatypes.Action(data=stacked[..., 0, :], valid=valids)
