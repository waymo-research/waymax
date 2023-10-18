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

from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf

from absl.testing import parameterized
from waymax import config as _config
from waymax import dataloader
from waymax import datatypes
from waymax.dynamics import bicycle_model
from waymax.utils import geometry
from waymax.utils import test_utils

TEST_DATA_PATH = test_utils.ROUTE_DATA_PATH

MAX_ACCEL = 6.0  # Units: m/s^2
MAX_STEER = 0.3  # Units: 1/m (curvature)


class BicycleModelTest(tf.test.TestCase, parameterized.TestCase):
  """Tests the action specs in the new random environment."""

  def setUp(self):
    super().setUp()
    self.sim_state = next(
        dataloader.simulator_state_generator(
            config=_config.DatasetConfig(
                path=TEST_DATA_PATH,
                max_num_objects=32,
                data_format=_config.DataFormat.TFRECORD,
            )
        )
    )

  @parameterized.product(
      batch_dims=(tuple(), (2, 1)),
      which_model=('InvertibleBicycleModel',),
      controlled_obj=(
          _config.ObjectType.SDC,
          _config.ObjectType.MODELED,
      ),
  )
  def test_dynamics_model_recovers_log_by_inverse_and_forward(
      self,
      batch_dims: Tuple[int, ...],
      which_model: str,
      controlled_obj: _config.ObjectType,
  ):
    # This test checks that forward and inverse are inverses of each other.
    # We first infer an action using inverse dynamics, and run the forward
    # update. We check that the forward update matches the original trajectory.
    # Read the initial state at t=0 from the dataset.
    state_t0 = next(
        dataloader.simulator_state_generator(
            config=_config.DatasetConfig(
                path=TEST_DATA_PATH,
                max_num_objects=32,
                batch_dims=batch_dims,
                data_format=_config.DataFormat.TFRECORD,
            )
        )
    )
    # Initialize the first 10 steps of sim_trajectory with log_trajectory.
    state_t10 = datatypes.update_state_by_log(state_t0, num_steps=10)
    # This test expects all objects to be updated.
    is_controlled = jnp.ones_like(
        datatypes.get_control_mask(state_t0.object_metadata, controlled_obj),
        dtype=jnp.bool_,
    )

    dynamics_model = getattr(bicycle_model, which_model)()
    # Run inverse dynamics on logged trajectory to infer actions.
    actions = dynamics_model.inverse(
        state_t10.log_trajectory, state_t0.object_metadata, timestep=10
    )
    # Apply inferred actions to sim trajectory to recover logged trajectory.
    new_traj = dynamics_model.forward(
        actions,
        trajectory=state_t10.sim_trajectory,
        reference_trajectory=state_t10.log_trajectory,
        is_controlled=is_controlled,
        timestep=10,
    )
    self.assertTupleEqual(actions.shape, batch_dims + (state_t0.num_objects,))
    self.assertAllClose(
        new_traj.x[..., :11], state_t10.log_trajectory.x[..., :11]
    )
    self.assertAllClose(
        new_traj.x[..., 12:],
        jnp.zeros_like(new_traj.x[..., 12:], dtype=jnp.float32),
    )
    # Check that running forward on the inferred action recovers the log
    # trajectory for both controlled and un-controlled objects.
    self.assertAllClose(
        new_traj.x[actions.valid[..., 0], 11],
        state_t10.log_trajectory.x[actions.valid[..., 0], 11],
        rtol=1e-3,
        atol=1e-3,
    )
    self.assertAllClose(
        new_traj.y[actions.valid[..., 0], 11],
        state_t10.log_trajectory.y[actions.valid[..., 0], 11],
        rtol=1e-3,
        atol=1e-3,
    )

  def test_dynamics_model_is_jitable(self):
    is_controlled = datatypes.get_control_mask(
        self.sim_state.object_metadata, _config.ObjectType.MODELED
    )
    dynamics_model = bicycle_model.InvertibleBicycleModel()
    actions = jax.jit(dynamics_model.inverse)(
        self.sim_state.log_trajectory,
        metadata=self.sim_state.object_metadata,
        timestep=10,
    )
    jax.jit(dynamics_model.forward)(
        actions,
        trajectory=self.sim_state.sim_trajectory,
        reference_trajectory=self.sim_state.log_trajectory,
        is_controlled=is_controlled,
        timestep=10,
    )

  @parameterized.parameters(((),), ((2, 1),))
  def test_dynamics_model_computes_accurate_states_from_test_data(
      self, batch_dims
  ):
    sim_state = next(
        dataloader.simulator_state_generator(
            config=_config.DatasetConfig(
                path=TEST_DATA_PATH,
                max_num_objects=32,
                batch_dims=batch_dims,
                data_format=_config.DataFormat.TFRECORD,
            )
        )
    )
    log_traj = sim_state.log_trajectory
    dynamics_model = bicycle_model.InvertibleBicycleModel(
        max_accel=MAX_ACCEL, max_steering=MAX_STEER, normalize_actions=True
    )

    # Note there are objects valid at timestep 18 but invalid at timestep 19
    # in this example.
    t = 18
    actions = dynamics_model.inverse(
        log_traj, sim_state.object_metadata, timestep=t
    )
    dt = dynamics_model._dt

    # The following code computes the expected action based on the equations
    # of the dynamics model. This checks for regression behavior and also
    # correct behavior on batched datatypes.
    # Shape: (..., num_objects=32)
    valid = jnp.logical_and(log_traj.valid[..., t], log_traj.valid[..., t + 1])
    # Shape: (..., num_objects=32, 1) for speed, yaw, real_yaw, accel, steering.
    speed = jnp.sqrt(
        log_traj.vel_x[..., t, None] ** 2 + log_traj.vel_y[..., t, None] ** 2
    )
    new_speed = jnp.sqrt(
        log_traj.vel_x[..., t + 1, None] ** 2
        + log_traj.vel_y[..., t + 1, None] ** 2
    )
    yaw0 = geometry.wrap_yaws(log_traj.yaw[..., t, None])
    yaw1 = geometry.wrap_yaws(log_traj.yaw[..., t + 1, None])
    real_yaw1 = jnp.arctan2(
        log_traj.vel_y[..., t + 1, None], log_traj.vel_x[..., t + 1, None]
    )
    real_yaw1 = jnp.where(
        jnp.abs(new_speed) <= bicycle_model._SPEED_LIMIT,
        yaw1,
        real_yaw1,
    )
    delta_yaw = geometry.wrap_yaws(real_yaw1 - yaw0)
    accel = (new_speed - speed) / dt
    steering = delta_yaw / (speed * dt + 0.5 * accel * dt**2)
    steering = jnp.where(
        jnp.abs(speed) < bicycle_model._SPEED_LIMIT, 0, steering
    )
    steering = jnp.where(
        jnp.abs(new_speed) < bicycle_model._SPEED_LIMIT, 0, steering
    )
    accel = jnp.clip(accel, -MAX_ACCEL, MAX_ACCEL) / MAX_ACCEL
    steering = jnp.clip(steering, -MAX_STEER, MAX_STEER) / MAX_STEER
    raw = jnp.concatenate([accel, steering], axis=-1)
    expect = jnp.where(valid[..., None], raw, 0.0)

    self.assertAllClose(actions.data, expect)
    self.assertAllClose(actions.valid, valid[..., None])

  def test_dynamics_model_computes_accurate_states_with_fixed_values(self):
    num_objects = 1
    config = _config.DatasetConfig(path='', max_num_objects=num_objects)
    test_state = test_utils.make_zeros_state(config)

    dynamics_model = bicycle_model.InvertibleBicycleModel(
        max_accel=MAX_ACCEL, max_steering=MAX_STEER, normalize_actions=True
    )
    sim_trajectory = test_state.sim_trajectory
    log_trajectory = test_state.log_trajectory
    action = jnp.tile(jnp.array([0.1, 0.1]), (num_objects, 1))
    actions = datatypes.Action(
        data=action.astype(jnp.float32),
        valid=jnp.ones_like(action, dtype=jnp.bool_)[..., 0:1],
    )
    new_traj = dynamics_model.forward(
        actions,
        sim_trajectory,
        log_trajectory,
        is_controlled=test_state.object_metadata.is_sdc,
        timestep=0,
    )
    self.assertAllClose(new_traj.x[0, 1], 0.003)
    self.assertAllClose(new_traj.y[0, 1], 0.0)
    self.assertAllClose(new_traj.vel_x[0, 1], 0.06)
    self.assertAllClose(new_traj.vel_y[0, 1], 0.0, atol=1e-5, rtol=1e-5)


class BicycleInverseTest(tf.test.TestCase):

  def test_acc_steering_inverse_computes_accurate_inverse_actions(self):
    """Test acceleration steering inverse dynamics for its correctness."""
    # This test checks the accuracy of inverse on random datatypes.
    # We first generate random actions (acceleration and steering) and
    # generate a trajectory using forward dynamics. We then compute the
    # inverse for each timestep and check that it matches the randomly
    # generated actions.

    dataset = test_utils.make_test_dataset()
    data_dict = next(dataset.as_numpy_iterator())
    sim_state_init = dataloader.simulator_state_from_womd_dict(
        data_dict, time_key='all'
    )
    # Use forward kinematics to calculate the trajectory with actions that
    # violate kinematics infeasibility metrics and actions that do not violate
    # kinematics infeasibility metrics. Assume initially we have x, y, yaw,
    # all equal to 0 and vel_x, vel_y all equal to 1.
    # Assume the max_acc = 6.0, and max_steering = 0.3.
    dt = 0.1
    total_steps = sim_state_init.sim_trajectory.x.shape[-1]
    num_agents = sim_state_init.sim_trajectory.x.shape[-2]

    # First, generate random actions for testing.
    accel = np.random.uniform(0.1, 5.0, size=total_steps - 1)
    steering = np.random.uniform(0.1, 0.25, size=total_steps - 1)
    traj_x, traj_y, traj_yaw, traj_vel_x, traj_vel_y = [0], [0], [0], [1], [1]
    traj_accel = []
    # Next, compute the entire trajectory state based on the equations of the
    # forward dynamics model. This code is equivalent to the equations used in
    # the dynamics implementation.
    for step_idx in range(total_steps - 1):
      accel_x = accel[step_idx] * np.cos(traj_yaw[step_idx])
      accel_y = accel[step_idx] * np.sin(traj_yaw[step_idx])
      traj_accel.append(np.sqrt(accel_x**2 + accel_y**2))
      new_x = traj_x[-1] + traj_vel_x[-1] * dt + 0.5 * accel_x * dt**2
      new_y = traj_y[-1] + traj_vel_y[-1] * dt + 0.5 * accel_y * dt**2
      vel = jnp.sqrt(traj_vel_x[-1] ** 2 + traj_vel_y[-1] ** 2)
      new_yaw = traj_yaw[-1] + steering[step_idx] * (
          vel * dt + 0.5 * accel[step_idx] * dt**2
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
    traj_valids = sim_state_init.sim_trajectory.valid * 0 + 1
    sim_state_init.sim_trajectory.x = jnp.repeat(
        traj_x, num_agents, -2
    ).reshape((num_agents, total_steps))
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

    # Finally, for each timestep, check that the computed inverse matches the
    # randomly generated accel and steering actions.
    for time_idx in range(total_steps - 1):
      actions = bicycle_model.compute_inverse(
          sim_state_init.sim_trajectory, time_idx, dt
      )
      self.assertEqual(actions.data.shape, (num_agents, 2))
      expected_action = jnp.concatenate(
          [
              jnp.array([traj_accel[time_idx]] * num_agents).reshape(
                  (num_agents, 1)
              ),
              jnp.array([steering[time_idx]] * num_agents).reshape(
                  (num_agents, 1)
              ),
          ],
          axis=-1,
      )
      self.assertAllClose(actions.data, expected_action, rtol=1e-3, atol=1e-3)

  def test_inverse_produces_correct_values(self):
    num_objects = 1
    config = _config.DatasetConfig(path='', max_num_objects=num_objects)
    test_state = test_utils.make_zeros_state(config)
    ones_traj = test_state.sim_trajectory.replace(
        x=jnp.ones_like(test_state.sim_trajectory.x),
        y=jnp.ones_like(test_state.sim_trajectory.x),
        vel_x=jnp.ones_like(test_state.sim_trajectory.x),
        vel_y=jnp.ones_like(test_state.sim_trajectory.x),
    )
    test_traj = datatypes.update_by_slice_in_dim(
        test_state.sim_trajectory, ones_traj, 1, 1, 50, axis=-1
    )
    test_traj = test_traj.replace(
        valid=jnp.ones_like(test_traj.valid, dtype=jnp.bool_)
    )
    actions = bicycle_model.compute_inverse(test_traj, 0, dt=0.1)
    self.assertAllClose(actions.data[0, 0], 14.142136)
    self.assertAllClose(actions.data[0, 1], 0.0)


if __name__ == '__main__':
  tf.test.main()
