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

"""Metrics relating to overlaps."""
import jax
from jax import numpy as jnp

from waymax import datatypes
from waymax.metrics import abstract_metric
from waymax.utils import geometry


class OverlapMetric(abstract_metric.AbstractMetric):
  """Overlap metric.

  This metric returns 1.0 if an object's bounding box is overlapping with
  that of another object.
  """

  @jax.named_scope('OverlapMetric.compute')
  def compute(
      self, simulator_state: datatypes.SimulatorState
  ) -> abstract_metric.MetricResult:
    current_object_state = datatypes.dynamic_slice(
        simulator_state.sim_trajectory,
        simulator_state.timestep,
        1,
        -1,
    )
    return self.compute_overlap(current_object_state)

  def compute_overlap(
      self, current_traj: datatypes.Trajectory
  ) -> abstract_metric.MetricResult:
    """Computes the overlap metric.

    Args:
      current_traj: Trajectory object containing current states of shape (...,
        num_objects, num_timesteps=1).

    Returns:
      A (..., num_objects) MetricResult.
    """
    traj_5dof = current_traj.stack_fields(['x', 'y', 'length', 'width', 'yaw'])
    # Shape: (..., num_objects, num_objects)
    pairwise_overlap = geometry.compute_pairwise_overlaps(traj_5dof[..., 0, :])

    # Remove overlaps with invalid objects
    # This is a no-op, but explicitly writing this since we want to
    # broadcast logical_and across agents, but the last dimension by default
    # corresponds to time.
    valid = current_traj.valid[..., 0:1]  # Shape: (..., num_objects, 1)
    pairwise_overlap = jnp.logical_and(pairwise_overlap, valid)
    num_overlap = jnp.sum(pairwise_overlap, axis=-2)
    overlap_indication = (num_overlap > 0).astype(jnp.float32)
    return abstract_metric.MetricResult.create_and_validate(
        overlap_indication, valid[..., 0]
    )
