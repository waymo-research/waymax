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

"""Tests for Waymax comfort metrics."""

from jax import numpy as jnp
import numpy as np
import tensorflow as tf

from absl.testing import parameterized
from waymax import dataloader
from waymax.metrics import comfort
from waymax.utils import geometry
from waymax.utils import test_utils

TEST_DATA_PATH = test_utils.ROUTE_DATA_PATH


class KinematicsInfeasibilityMetricTest(
    tf.test.TestCase, parameterized.TestCase
):

  @parameterized.parameters((-0.1, True), (0.1, False))
  def test_kinematics_infeasibility_metric_runs_from_synthetic_data(
      self, delta_vel, kim_value
  ):
    dataset = test_utils.make_test_dataset()
    data_dict = next(dataset.as_numpy_iterator())
    sim_state_init = dataloader.simulator_state_from_womd_dict(
        data_dict, time_key='all'
    )
    dt, max_acc, max_steering = 0.1, 10.4, 0.3
    kim = comfort.KinematicsInfeasibilityMetric(dt, max_acc, max_steering)

    # Assume a perfect circular trajectory with 0 acceleration
    # to violate the curvature limit, we can use a velocity that is
    # smaller than (traj_yaw[t]-traj_yaw[t-1]) / (max_steering * dt)
    # which is v_min = (2 * pi / total_steps) / (max_steering * dt).
    radius = 1 / max_steering
    total_steps = sim_state_init.sim_trajectory.x.shape[-1]
    num_agents = sim_state_init.sim_trajectory.x.shape[-2]
    traj_vel = 2 * jnp.pi / total_steps / max_steering / dt + delta_vel
    traj_yaw = (
        jnp.linspace(0, jnp.pi * 2, total_steps + 1)[:-1] + jnp.pi / 2.0
    ).reshape((1, total_steps))
    traj_vel_x = traj_vel * jnp.cos(traj_yaw)
    traj_vel_y = traj_vel * jnp.sin(traj_yaw)
    traj_valids = jnp.ones_like(sim_state_init.sim_trajectory.valid)
    traj_xs, traj_ys = [radius], [0.0]
    for i in range(total_steps - 1):
      traj_xs.append(traj_xs[-1] + float(traj_vel_x[0, i]) * dt)
      traj_ys.append(traj_ys[-1] + float(traj_vel_y[0, i]) * dt)
    traj_x = jnp.array(traj_xs).reshape((1, total_steps))
    traj_y = jnp.array(traj_ys).reshape((1, total_steps))
    sim_state_init.timestep = 11
    # Repeat this trajectory for all agents.
    # Shape: (..., num_objects, num_timesteps).
    repeated_objects = jnp.repeat(traj_x, num_agents, -2).reshape(
        (num_agents, total_steps)
    )
    sim_state_init.sim_trajectory.x = repeated_objects
    sim_state_init.sim_trajectory.y = jnp.repeat(
        traj_y, num_agents, -2
    ).reshape((num_agents, total_steps))
    sim_state_init.sim_trajectory.yaw = jnp.repeat(
        traj_yaw, num_agents, -2
    ).reshape((num_agents, total_steps))
    sim_state_init.sim_trajectory.vel_x = jnp.repeat(
        traj_vel_x, num_agents, -2
    ).reshape((num_agents, total_steps))
    sim_state_init.sim_trajectory.vel_y = jnp.repeat(
        traj_vel_y, num_agents, -2
    ).reshape((num_agents, total_steps))
    sim_state_init.sim_trajectory.valid = traj_valids
    sim_state_init.sim_trajectory.validate()
    expected_valids = traj_valids[..., sim_state_init.timestep]
    result = kim.compute(sim_state_init)
    self.assertEqual(result.shape, (128,))
    self.assertAllClose(result.value, jnp.array([1.0 * kim_value] * 128))
    self.assertAllEqual(result.valid, expected_valids)

  def test_kinematics_infeasibility_metric_runs_from_large_data(self):
    dataset = test_utils.make_test_dataset()
    data_dict = next(dataset.as_numpy_iterator())
    sim_state_init = dataloader.simulator_state_from_womd_dict(
        data_dict, time_key='all'
    )
    # Use forward kinematics to calculate the trajectory with actions that
    # violate kinematics infeasibility metrics and actions that do not violate
    # kinematics infeasibility metrics. Assume initially we have x, y, yaw,
    # all equal to 0 and vel_x, vel_y all equal to 1.
    # Assume the max_acc = 10.4, and max_steering = 0.3.
    dt, max_acc, max_steering = 0.1, 10.4, 0.3
    kim = comfort.KinematicsInfeasibilityMetric(dt, max_acc, max_steering)
    total_steps = sim_state_init.sim_trajectory.x.shape[-1]
    num_agents = sim_state_init.sim_trajectory.x.shape[-2]
    accel = np.random.uniform(0.1, 7.0, size=total_steps - 1)
    steering = np.random.uniform(0.1, 0.4, size=total_steps - 1)
    traj_x, traj_y, traj_yaw, traj_vel_x, traj_vel_y = [0], [0], [0], [1], [1]
    for step_idx in range(total_steps - 1):
      accel_x = accel[step_idx] * np.cos(traj_yaw[step_idx])
      accel_y = accel[step_idx] * np.sin(traj_yaw[step_idx])
      new_x = traj_x[-1] + traj_vel_x[-1] * dt + 0.5 * accel_x * dt**2
      new_y = traj_y[-1] + traj_vel_y[-1] * dt + 0.5 * accel_y * dt**2
      vel = jnp.sqrt(traj_vel_x[-1] ** 2 + traj_vel_y[-1] ** 2)
      new_yaw = geometry.wrap_yaws(
          traj_yaw[-1]
          + steering[step_idx] * (vel * dt + 0.5 * accel[step_idx] * dt**2)
      )
      new_vel = vel + accel[step_idx] * dt
      new_vel_x = new_vel * np.cos(new_yaw)
      new_vel_y = new_vel * np.sin(new_yaw)
      traj_x.append(new_x)
      traj_y.append(new_y)
      traj_yaw.append(new_yaw)
      traj_vel_x.append(new_vel_x)
      traj_vel_y.append(new_vel_y)
    traj_x = jnp.array(traj_x).reshape((1, total_steps))
    traj_y = jnp.array(traj_y).reshape((1, total_steps))
    traj_yaw = jnp.array(traj_yaw).reshape((1, total_steps))
    traj_vel_x = jnp.array(traj_vel_x).reshape((1, total_steps))
    traj_vel_y = jnp.array(traj_vel_y).reshape((1, total_steps))
    traj_valids = jnp.ones_like(sim_state_init.sim_trajectory.valid)
    # Repeat this trajectory for all agents.
    # Shape: (..., num_objects, num_timesteps).
    repeated_objects = jnp.repeat(traj_x, num_agents, -2).reshape(
        (num_agents, total_steps)
    )

    sim_state_init.sim_trajectory.x = repeated_objects
    sim_state_init.sim_trajectory.y = jnp.repeat(
        traj_y, num_agents, -2
    ).reshape((num_agents, total_steps))
    sim_state_init.sim_trajectory.yaw = jnp.repeat(
        traj_yaw, num_agents, -2
    ).reshape((num_agents, total_steps))
    sim_state_init.sim_trajectory.vel_x = jnp.repeat(
        traj_vel_x, num_agents, -2
    ).reshape((num_agents, total_steps))
    sim_state_init.sim_trajectory.vel_y = jnp.repeat(
        traj_vel_y, num_agents, -2
    ).reshape((num_agents, total_steps))
    sim_state_init.sim_trajectory.valid = traj_valids
    sim_state_init.sim_trajectory.validate()
    all_kim_values_gt = jnp.logical_or(
        jnp.abs(accel) > max_acc + comfort._KIM_EPSILON,
        jnp.abs(steering) > max_steering + comfort._KIM_EPSILON,
    )
    for time_idx in range(1, total_steps):
      sim_state_init.timestep = time_idx
      result = kim.compute(sim_state_init)
      self.assertEqual(result.shape, (num_agents,))
      self.assertAllClose(
          result.value,
          jnp.array([1.0 * all_kim_values_gt[time_idx - 1]] * num_agents),
      )
      self.assertAllEqual(result.valid, traj_valids[..., time_idx])


if __name__ == '__main__':
  tf.test.main()
