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

"""Metrics relating to comfort."""
import jax
from jax import numpy as jnp

from waymax import datatypes
from waymax.dynamics import bicycle_model
from waymax.metrics import abstract_metric

# A constant term to avoid numerical instability in the calculation of
# kinematics infeasibility metrics.
_KIM_EPSILON = 1e-3


class KinematicsInfeasibilityMetric(abstract_metric.AbstractMetric):
  """Kinematics infeasibility metric.

  The metric uses continuous acceleration, steering inverse dynamics model to
  estimate the required acceleration and steering curvature to complete the
  transition. This metric returns 1 if the magnitude of the estimated
  acceleration and/or steering curvature exceeds the max feasibility magnitude,
  otherwise returns 0.0.
  """

  def __init__(
      self, dt: float = 0.1, max_acc: float = 10.4, max_steering: float = 0.3
  ):
    """Initializes the kinematics infeasibility metrics.


    Args:
      dt: The time gap length per environment step.
      max_acc: Maximum value of the magnitude of acceleration, which is the
        change of the speed per second at the current step.
      max_steering: Maximum value of the magnitude of steering, which is the
        turning curvature of the vehicle. The turning curvature is the inverse
        of the turning radius, which is determined by the steering wheel angle.
    """
    self._dt = dt
    self._max_acc = max_acc
    self._max_steering = max_steering

  @jax.named_scope('KinematicsInfeasibilityMetric.compute')
  def compute(
      self, simulator_state: datatypes.SimulatorState
  ) -> abstract_metric.MetricResult:
    """Computes the kinematics infeasibility metric.

    The metrics detects whether the acceleration, steering curvature at step
    simulator_state.timestep violates the allowed magnitude ranges. The
    acceleration, steering curvature corresponds to the transition from
    simulator_state.timestep-1 to simulator_state.timestep. The value of
    simulator_state.timestep means that all states are updated up to the step
    of simulator_state.timestep and simulator_state.timestep+1 is still invalid.
    Therefore, we need to test whether the transition from
    simulator_state.timestep-1 to simulator_state.timestep is kinematically
    feasible.

    Args:
      simulator_state: The current simulator state of shape of shape (...).

    Returns:
      A (..., num_objects) MetricResult containing the metric result, with
        values in either 1 or 0, where 1.0 indicates that the current step
        violates the kinematic infeasibility metrics, and 0.0 indicates that the
        current step does not violate the kinematic infeasibility metrics.
    """
    return self.compute_kinematics_infeasibility(
        simulator_state.sim_trajectory, simulator_state.timestep
    )

  def compute_kinematics_infeasibility(
      self, traj: datatypes.Trajectory, timestep: jax.Array
  ) -> abstract_metric.MetricResult:
    """Computes the kinematics infeasibility metric.


    Args:
      traj: The sim_trajectory of all steps, of shape (...,  num_objects,
        num_timesteps).
      timestep: The current simulator timestep at which the kinematics
        infeasibility metric is calculated at.

    Returns:
      A (..., num_objects) MetricResult containing the metric result.
    """
    actions = bicycle_model.compute_inverse(traj, timestep - 1, self._dt)
    action_array = actions.data
    accel = action_array[..., 0]
    steering = action_array[..., 1]
    # Add an epsilon term to avoid numerical instability.
    kim_infeasibility_bool = jnp.logical_or(
        jnp.abs(accel) > self._max_acc + _KIM_EPSILON,
        jnp.abs(steering) > self._max_steering + _KIM_EPSILON,
    )
    kim_infeasibility = 1.0 * kim_infeasibility_bool
    return abstract_metric.MetricResult.create_and_validate(
        kim_infeasibility, actions.valid[..., 0]
    )
