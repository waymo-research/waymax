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

"""Metrics relating to imitation."""
import jax
from jax import numpy as jnp

from waymax import datatypes
from waymax.metrics import abstract_metric


class LogDivergenceMetric(abstract_metric.AbstractMetric):
  """Log divergence metric.

  This metric returns the L2 distance between the controlled object's XY
  location and its position in the logged history at the same timestep.
  """

  @jax.named_scope('LogDivergenceMetric.compute')
  def compute(
      self, simulator_state: datatypes.SimulatorState
  ) -> abstract_metric.MetricResult:
    """Computes log divergence by fetching correct arguments.

    Args:
      simulator_state: The current simulator state.

    Returns:
      A (..., num_objects) MetricResult containing the metric result, with L2
      values representing the distance of each object from the log to the
      simulated trajectory at the current timestep.
    """
    current_object_state = datatypes.dynamic_slice(
        simulator_state.sim_trajectory,
        simulator_state.timestep,
        1,
        -1,
    )
    current_log_state = datatypes.dynamic_slice(
        simulator_state.log_trajectory,
        simulator_state.timestep,
        1,
        -1,
    )
    result = self.compute_log_divergence(
        current_object_state.xy, current_log_state.xy
    )
    valid = current_object_state.valid & current_log_state.valid
    return abstract_metric.MetricResult.create_and_validate(
        result[..., 0], valid[..., 0]
    )

  @classmethod
  def compute_log_divergence(
      cls, object_xy: jax.Array, log_xy: jax.Array
  ) -> jax.Array:
    """Computes the L2 distance between `object_xy` and `log_xy`.

    Args:
      object_xy: XY coordinates of current vehicles for the current timestep, of
        shape (..., num_objects, num_timesteps, 2).
      log_xy: XY coordinates of logged vehicles for the current timestep, of
        shape (..., num_objects, num_timesteps, 2).

    Returns:
      A (..., num_objects, num_timesteps) array containing the metric result of
        the same shape as the input trajectories.
    """
    return jnp.linalg.norm(object_xy - log_xy, axis=-1)
