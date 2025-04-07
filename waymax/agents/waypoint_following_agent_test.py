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

"""Tests for waypoint_following_agent."""

import jax
from jax import numpy as jnp
import tensorflow as tf

from absl.testing import parameterized
from waymax import config as _config
from waymax import dataloader
from waymax import datatypes
from waymax.agents import constant_speed
from waymax.agents import waypoint_following_agent
from waymax.utils import test_utils

TEST_DATA_PATH = test_utils.ROUTE_DATA_PATH


def make_sim_traj(
    ref_traj: datatypes.Trajectory, position: jax.Array, speed: float = 1.0
) -> datatypes.Trajectory:
  new_x = jnp.ones_like(ref_traj.x) * position[..., 0]
  new_y = jnp.ones_like(ref_traj.y) * position[..., 1]
  vel_x = speed * jnp.cos(ref_traj.yaw)
  vel_y = speed * jnp.sin(ref_traj.yaw)
  new_traj = ref_traj.replace(x=new_x, y=new_y, vel_x=vel_x, vel_y=vel_y)
  new_traj.validate()
  return datatypes.dynamic_slice(new_traj, 0, 1, axis=-1)


class WaypointFollowingAgentTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    test_traj = datatypes.Trajectory(
        x=jnp.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
        y=jnp.array([-1.0, 0.0, 0.0, 0.0, 1.0, 2.0]),
        z=jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        vel_x=jnp.array([15.0, 10.0, 10.0, 10.0, 10.0, 0.0]),
        vel_y=jnp.array([15.0, 10.0, 10.0, 10.0, 10.0, 0.0]),
        yaw=jnp.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
        timestamp_micros=jnp.array(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.int32
        ),
        valid=jnp.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=bool),
        length=jnp.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
        width=jnp.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
        height=jnp.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
    )
    self.test_traj = jax.tree.map(lambda x: x[jnp.newaxis], test_traj)
    test_traj_with_invalid = test_traj.replace(
        x=jnp.array([0.0, 1.0, -1.0, -1.0, 4.0, 5.0]),
        y=jnp.array([-1.0, 0.0, -1.0, -1.0, 1.0, 2.0]),
        z=jnp.array([0.0, 0.0, -1.0, -1.0, 0.0, 0.0]),
        valid=jnp.array([1.0, 1.0, 0.0, 0.0, 1.0, 1.0], dtype=bool),
    )
    self.test_traj_with_invalid = jax.tree.map(
        lambda x: x[jnp.newaxis], test_traj_with_invalid
    )
    self.config = _config.DatasetConfig(
        path=TEST_DATA_PATH,
        max_num_objects=8,
        data_format=_config.DataFormat.TFRECORD,
    )
    self.state_t0 = next(dataloader.simulator_state_generator(self.config))

    self.config_batched = _config.DatasetConfig(
        path=TEST_DATA_PATH,
        max_num_objects=8,
        batch_dims=(4,),
        data_format=_config.DataFormat.TFRECORD,
    )
    self.state_batched = next(
        dataloader.simulator_state_generator(self.config_batched)
    )

  def test_invalidates_at_end_of_trajectory(self):
    new_speed = jnp.ones((1,), dtype=jnp.float32) * 1000.0
    new_valid = jnp.ones_like(new_speed, dtype=jnp.bool_)
    with self.subTest('valid trajectory'):
      cur_sim_traj = make_sim_traj(
          self.test_traj, self.test_traj.xyz[0, -1], 1000.0
      )
      next_traj = constant_speed.ConstantSpeedPolicy(
          speed=10.0
      )._get_next_trajectory_by_projection(
          self.test_traj,
          cur_sim_traj=cur_sim_traj,
          new_speed=new_speed,
          new_speed_valid=new_valid,
      )
      self.assertFalse(next_traj.valid[0])

    with self.subTest('invalid trajectory'):
      # Both of these cases should be invalidated.

      # Current log state valid, next state invalid.
      cur_sim_traj = make_sim_traj(
          self.test_traj, self.test_traj.xyz[0, 1], 1000.0
      )
      next_traj = constant_speed.ConstantSpeedPolicy(
          speed=10.0
      )._get_next_trajectory_by_projection(
          self.test_traj_with_invalid,
          cur_sim_traj=cur_sim_traj,
          new_speed=new_speed,
          new_speed_valid=new_valid,
      )
      self.assertFalse(next_traj.valid[0])

      # Current log state invalid, next state valid.
      cur_sim_traj = make_sim_traj(
          self.test_traj, self.test_traj.xyz[0, 3], 1000.0
      )
      next_traj = constant_speed.ConstantSpeedPolicy(
          speed=10.0
      )._get_next_trajectory_by_projection(
          self.test_traj_with_invalid,
          cur_sim_traj=cur_sim_traj,
          new_speed=new_speed,
          new_speed_valid=new_valid,
      )
      self.assertFalse(next_traj.valid[0])

  def test_next_speed_is_set(self):
    new_speed = jnp.ones((1,), dtype=jnp.float32) * 4.5
    new_valid = jnp.ones_like(new_speed, dtype=jnp.bool_)
    cur_sim_traj = make_sim_traj(self.test_traj, self.test_traj.xyz[0, 1], 1.0)
    next_traj = constant_speed.ConstantSpeedPolicy(
        speed=13.0
    )._get_next_trajectory_by_projection(
        self.test_traj,
        cur_sim_traj=cur_sim_traj,
        new_speed=new_speed,
        new_speed_valid=new_valid,
    )
    result_speed = jnp.linalg.norm(next_traj.vel_xy[0, 0], keepdims=True)
    self.assertAllClose(result_speed, new_speed)

  def test_zero_velocity_does_not_move(self):
    new_speed = jnp.ones((1,), dtype=jnp.float32) * 0.0
    new_valid = jnp.ones_like(new_speed, dtype=jnp.bool_)
    cur_sim_traj = make_sim_traj(self.test_traj, self.test_traj.xyz[0, 1], 0.0)
    next_traj = constant_speed.ConstantSpeedPolicy(
        speed=13.0
    )._get_next_trajectory_by_projection(
        self.test_traj,
        cur_sim_traj=cur_sim_traj,
        new_speed=new_speed,
        new_speed_valid=new_valid,
    )
    next_xyz = next_traj.xyz[0, 0]
    self.assertAllClose(next_xyz, self.test_traj.xyz[0, 1])

  def test_high_velocity_reaches_final_waypoint(self):
    new_speed = jnp.ones((1,), dtype=jnp.float32) * 0.0
    new_valid = jnp.ones_like(new_speed, dtype=jnp.bool_)
    cur_sim_traj = make_sim_traj(
        self.test_traj, self.test_traj.xyz[0, 1], 1000.0
    )
    next_traj = constant_speed.ConstantSpeedPolicy(
        speed=13.0
    )._get_next_trajectory_by_projection(
        self.test_traj,
        cur_sim_traj=cur_sim_traj,
        new_speed=new_speed,
        new_speed_valid=new_valid,
        dt=0.1,
    )
    next_xyz = next_traj.xyz[0, 0]
    self.assertAllClose(next_xyz, self.test_traj.xyz[0, 5])

  def test_runs_e2e_on_real_data(self):
    with self.subTest('TestUnbatched'):
      next_traj = constant_speed.ConstantSpeedPolicy(
          speed=13.0
      ).update_trajectory(self.state_t0)
      next_traj.validate()
      self.assertEqual(next_traj.shape, (self.config.max_num_objects, 1))

    with self.subTest('TestBatched'):
      next_traj = constant_speed.ConstantSpeedPolicy(
          speed=13.0
      ).update_trajectory(self.state_batched)
      next_traj.validate()
      self.assertEqual(
          next_traj.shape,
          self.config_batched.batch_dims
          + (self.config_batched.max_num_objects, 1),
      )


class IDMRoutePolicyTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    test_traj = datatypes.Trajectory(
        x=jnp.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
        y=jnp.array([-1.0, 0.0, 0.0, 0.0, 1.0, 2.0]),
        z=jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        vel_x=jnp.array([10.0, 10.0, 10.0, 10.0, 10.0, 0.0]),
        vel_y=jnp.array([10.0, 10.0, 10.0, 10.0, 10.0, 0.0]),
        yaw=jnp.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
        timestamp_micros=jnp.array([0, 0, 0, 0, 0, 0], dtype=jnp.int32),
        valid=jnp.array([True, True, True, True, True, True]),
        length=jnp.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
        width=jnp.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
        height=jnp.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
    )
    self.test_traj = jax.tree.map(lambda x: x[jnp.newaxis], test_traj)

    self.config = _config.DatasetConfig(
        path=TEST_DATA_PATH,
        max_num_objects=8,
        data_format=_config.DataFormat.TFRECORD,
    )
    self.state_t0 = next(dataloader.simulator_state_generator(self.config))

    self.config_batched = _config.DatasetConfig(
        path=TEST_DATA_PATH,
        max_num_objects=8,
        batch_dims=(4,),
        data_format=_config.DataFormat.TFRECORD,
    )
    self.state_batched = next(
        dataloader.simulator_state_generator(self.config_batched)
    )

  def test_lead_distance(self):
    agent_future = jnp.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]])
    collision_indicator = jnp.array([False, False, True, True])
    result = waypoint_following_agent.IDMRoutePolicy()._compute_lead_distance(
        agent_future, collision_indicator
    )
    self.assertEqual(result, 2.0)

  def test_lead_velocity(self):
    future_speeds = jnp.array([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]])
    collision_per_agent = jnp.array([
        [False, False, False, True],
        [False, False, True, False],
        [False, True, True, False],
    ])
    result = waypoint_following_agent.IDMRoutePolicy()._compute_lead_velocity(
        future_speeds, collision_per_agent
    )
    self.assertEqual(result, 9.0)

  def test_decelerates_near_collision(self):
    # Create a test with a close collision, causing the vehicle to output
    # the max deceleration.
    obj_a = self.test_traj  # (1, 6)
    obj_b = datatypes.Trajectory(
        x=jnp.array([3.0]),
        y=jnp.array([0.0]),
        z=jnp.array([0.0]),
        vel_x=jnp.array([1.0]),
        vel_y=jnp.array([1.0]),
        yaw=jnp.array([0.0]),
        timestamp_micros=jnp.array([0]),
        valid=jnp.array([True]),
        length=jnp.array([0.1]),
        width=jnp.array([0.1]),
        height=jnp.array([0.1]),
    )
    obj_b = jax.tree_util.tree_map(
        lambda x: jnp.repeat(x[:, jnp.newaxis], obj_a.shape[-1], axis=-1), obj_b
    )
    objects = jax.tree_util.tree_map(
        lambda x, y: jnp.concatenate([x, y], axis=-2), obj_a, obj_b
    )
    cur_speed = jnp.array([10.0, 10.0])
    cur_position = jax.tree_util.tree_map(lambda x: x[..., :1], objects)
    max_accel = 1.13
    max_decel = 1.78
    delta = 4.0
    desired_vel = 30.0
    result = waypoint_following_agent.IDMRoutePolicy(
        max_accel=max_accel,
        max_decel=max_decel,
        desired_vel=desired_vel,
        min_spacing=1.0,
        safe_time_headway=1.0,
        max_lookahead=6,
        delta=delta,
        lookahead_from_current_position=False,
        invalidate_on_end=True,
    )._get_accel(objects, objects.xyz[:, 0, :], cur_speed, cur_position)
    # First agent should yield to second agent.
    # Second agent for free-road behavior.
    free_road_speed = max_accel * (1 - (cur_speed[0] / desired_vel) ** delta)
    self.assertLess(result[0], 0)
    self.assertAllClose(result[1], free_road_speed)

  @parameterized.parameters((1.0, 10.0, 30.0), (2.0, 10.0, 15.0))
  def test_free_road_behavior(self, max_accel, cur_speed, desired_speed):
    # Create a test with a no collisions, causing the vehicle to behave
    # according to the "free-road behavior", defined as:
    # accel = max_a * (1 - (v / v0)^delta))
    waypoints = self.test_traj
    delta = 4.0
    cur_position = jax.tree_util.tree_map(lambda x: x[..., :1], waypoints)
    result = waypoint_following_agent.IDMRoutePolicy(
        max_accel=max_accel,
        max_decel=max_accel,
        desired_vel=desired_speed,
        max_lookahead=6,
        delta=delta,
    )._get_accel(
        waypoints, waypoints.xyz[:, 0, :], jnp.array([cur_speed]), cur_position
    )

    expected_result = max_accel * (1 - (cur_speed / desired_speed) ** delta)
    self.assertAllClose(result, [expected_result])

  def test_runs_e2e_on_real_data(self):
    with self.subTest('TestUnbatched'):
      next_traj = waypoint_following_agent.IDMRoutePolicy().update_trajectory(
          self.state_t0
      )
      next_traj.validate()
      self.assertEqual(next_traj.shape, (self.config.max_num_objects, 1))

    with self.subTest('TestBatched'):
      next_traj = waypoint_following_agent.IDMRoutePolicy().update_trajectory(
          self.state_batched
      )
      next_traj.validate()
      self.assertEqual(
          next_traj.shape,
          self.config_batched.batch_dims
          + (self.config_batched.max_num_objects, 1),
      )


class UtilFunctionsTest(tf.test.TestCase, parameterized.TestCase):

  def test_add_headway_points(self):
    traj = datatypes.Trajectory(
        x=jnp.array([0.0, 1.0, 2.0]),
        y=jnp.array([0.0, 1.0, 2.0]),
        z=jnp.array([0.0, 1.0, 2.0]),
        vel_x=jnp.array([0.0, 1.0, 2.0]),
        vel_y=jnp.array([0.0, 1.0, 2.0]),
        yaw=jnp.array([1.0, 1.0, 0.0]),
        valid=jnp.array([True, True, True]),
        timestamp_micros=jnp.array([1, 1, 1]),
        length=jnp.array([1, 1, 1], dtype=jnp.float32),
        width=jnp.array([1, 1, 1], dtype=jnp.float32),
        height=jnp.array([1, 1, 1], dtype=jnp.float32),
    )
    traj = jax.tree.map(lambda x: x[jnp.newaxis], traj)

    new_traj = waypoint_following_agent._add_headway_waypoints(
        traj, distance=2.0, num_points=2
    )
    expected_traj = datatypes.Trajectory(
        x=jnp.array([0.0, 1.0, 2.0, 3.0, 4.0]),
        y=jnp.array([0.0, 1.0, 2.0, 2.0, 2.0]),
        z=jnp.array([0.0, 1.0, 2.0, 2.0, 2.0]),
        vel_x=jnp.array([0.0, 1.0, 2.0, 2.0, 2.0]),
        vel_y=jnp.array([0.0, 1.0, 2.0, 2.0, 2.0]),
        yaw=jnp.array([1.0, 1.0, 0.0, 0.0, 0.0]),
        valid=jnp.array([True, True, True, True, True]),
        timestamp_micros=jnp.array([1, 1, 1, 1, 1]),
        length=jnp.array([1, 1, 1, 1, 1]),
        width=jnp.array([1, 1, 1, 1, 1]),
        height=jnp.array([1, 1, 1, 1, 1]),
    )
    expected_traj = jax.tree.map(lambda x: x[jnp.newaxis], expected_traj)

    traj_7dof = new_traj.stack_fields(
        ['x', 'y', 'vel_x', 'vel_y', 'length', 'width', 'yaw']
    )
    exp_7dof = expected_traj.stack_fields(
        ['x', 'y', 'vel_x', 'vel_y', 'length', 'width', 'yaw']
    )
    self.assertAllClose(traj_7dof, exp_7dof)


if __name__ == '__main__':
  tf.test.main()
