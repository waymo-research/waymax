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

"""Tests for Waymax route metrics."""

from typing import Optional

import jax
from jax import numpy as jnp
import tensorflow as tf

from absl.testing import parameterized
from waymax import config as _config
from waymax import dataloader
from waymax import datatypes
from waymax.metrics import route
from waymax.utils import test_utils

ROUTE_DATA_PATH = test_utils.ROUTE_DATA_PATH
ROUTE_NUM_PATHS = test_utils.ROUTE_NUM_PATHS
ROUTE_NUM_POINTS_PER_PATH = test_utils.ROUTE_NUM_POINTS_PER_PATH


class ProgressionMetricTest(tf.test.TestCase, parameterized.TestCase):

  def test_raises_error_if_no_sdc_paths(self):
    traj = test_utils.create_test_trajectory_from_position()
    sim_state = create_sim_state_from_trajectories_and_paths(
        sim_trajectory=traj, log_trajectory=traj, sdc_paths=None
    )
    with self.assertRaises(ValueError):
      route.ProgressionMetric().compute(sim_state)

  def test_returns_invalid_if_sdc_traj_is_invalid(self):
    sim_traj = test_utils.create_test_trajectory_from_position(
        position=[1.0, 1.0, 1.0]
    )
    sim_traj = sim_traj.replace(
        valid=jnp.zeros_like(sim_traj.valid, dtype=jnp.bool_)
    )

    log_traj_xyz = jnp.array(
        [[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]], dtype=jnp.float32
    )
    log_traj = create_test_trajectory_from_positions(log_traj_xyz)

    num_paths = 1
    num_points_per_path = 1
    paths_x = jnp.ones((num_paths, num_points_per_path))
    valid = jnp.ones_like(paths_x, dtype=jnp.bool_)
    on_route = jnp.ones((num_paths, 1), dtype=jnp.bool_)
    arc_length = jnp.ones_like(paths_x)
    paths = create_paths(
        paths_x, paths_x, paths_x, valid, on_route, arc_length=arc_length
    )

    sim_state = create_sim_state_from_trajectories_and_paths(
        sim_trajectory=sim_traj, log_trajectory=log_traj, sdc_paths=paths
    )
    result = route.ProgressionMetric().compute(sim_state)
    self.assertEqual(result.value, 0.0)
    self.assertFalse(result.valid)

  def test_returns_invalid_if_no_valid_paths(self):
    sim_traj = test_utils.create_test_trajectory_from_position(
        position=[1.0, 1.0, 1.0]
    )

    log_traj_xyz = jnp.array(
        [[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]], dtype=jnp.float32
    )
    log_traj = create_test_trajectory_from_positions(log_traj_xyz)

    # Create one invalid on-route path.
    num_paths = 1
    num_points_per_path = 1
    paths_x = jnp.ones((num_paths, num_points_per_path))
    valid = jnp.zeros((num_paths, num_points_per_path), dtype=jnp.bool_)
    on_route = jnp.ones((num_paths, 1), dtype=jnp.bool_)
    arc_length = jnp.ones_like(paths_x)
    paths = create_paths(
        paths_x, paths_x, paths_x, valid, on_route, arc_length=arc_length
    )

    sim_state = create_sim_state_from_trajectories_and_paths(
        sim_trajectory=sim_traj, log_trajectory=log_traj, sdc_paths=paths
    )
    result = route.ProgressionMetric().compute(sim_state)
    self.assertEqual(result.value, 0.0)
    self.assertFalse(result.valid)

  def test_returns_invalid_if_no_valid_on_route_paths(self):
    sim_traj = test_utils.create_test_trajectory_from_position(
        position=[1.0, 1.0, 1.0]
    )

    log_traj_xyz = jnp.array(
        [[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]], dtype=jnp.float32
    )
    log_traj = create_test_trajectory_from_positions(log_traj_xyz)

    # Create an invalid on-route path and a valid off-route path.
    num_paths = 2
    num_points_per_path = 1
    paths_x = jnp.ones((num_paths, num_points_per_path))
    valid = jnp.array([[False], [True]], dtype=jnp.bool_)
    on_route = jnp.array([[True], [False]], dtype=jnp.bool_)
    arc_length = jnp.ones_like(paths_x)
    paths = create_paths(
        paths_x, paths_x, paths_x, valid, on_route, arc_length=arc_length
    )

    sim_state = create_sim_state_from_trajectories_and_paths(
        sim_trajectory=sim_traj, log_trajectory=log_traj, sdc_paths=paths
    )
    result = route.ProgressionMetric().compute(sim_state)
    self.assertEqual(result.value, 0.0)
    self.assertFalse(result.valid)

  @parameterized.named_parameters(
      ('start', 0.0, 0.0),
      ('middle', 1.0, 0.5),
      ('end', 2.0, 1.0),
      ('beyond_end', 4.0, 2.0),
  )
  def test_returns_expected_progress(self, sdc_x, expected_progress):
    sim_traj = test_utils.create_test_trajectory_from_position(
        position=[sdc_x, 0.0, 0.0]
    )

    log_traj_xyz = jnp.array(
        [[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]], dtype=jnp.float32
    )
    log_traj = create_test_trajectory_from_positions(log_traj_xyz)

    # Path 1 has arc_length = [0, 1, 2, 3, 4, 5] and is the reference path
    # because it is closest to the SDC sim trajectory.
    path1_arc_length = jnp.arange(0, 6, dtype=jnp.float32)
    path1_y = jnp.zeros_like(path1_arc_length)
    # Path 2 has arc_length = [0, 2, 4, 6, 8, 10] and is not the reference path
    # because it is farther from the SDC sim trajectory than path 1.
    path2_arc_length = 10.0 * path1_arc_length
    path2_y = jnp.zeros_like(path2_arc_length)

    paths_arc_length = jnp.stack([path1_arc_length, path2_arc_length], axis=0)
    paths_y = jnp.stack([path1_y, path2_y], axis=0)
    paths_z = jnp.zeros_like(paths_arc_length)
    valid = jnp.ones_like(paths_arc_length, dtype=jnp.bool_)
    on_route = jnp.ones((2, 1), dtype=jnp.bool_)
    paths = create_paths(
        paths_arc_length,
        paths_y,
        paths_z,
        valid,
        on_route,
        arc_length=paths_arc_length,
    )

    sim_state = create_sim_state_from_trajectories_and_paths(
        sim_trajectory=sim_traj, log_trajectory=log_traj, sdc_paths=paths
    )
    result = route.ProgressionMetric().compute(sim_state)
    self.assertEqual(result.value, expected_progress)
    self.assertTrue(result.valid)

  def test_log_data(self):
    batch_dims = (1, 2)
    config = _config.DatasetConfig(
        path=ROUTE_DATA_PATH,
        data_format=_config.DataFormat.TFRECORD,
        include_sdc_paths=True,
        max_num_rg_points=30000,
        batch_dims=batch_dims,
        num_paths=ROUTE_NUM_PATHS,
        num_points_per_path=ROUTE_NUM_POINTS_PER_PATH,
    )
    data_iter = dataloader.simulator_state_generator(config=config)
    state_t0 = next(data_iter)
    expected_valid = jnp.ones(shape=batch_dims, dtype=jnp.bool_)
    with self.subTest('start_position'):
      progress_t0 = route.ProgressionMetric().compute(state_t0)
      self.assertAllEqual(progress_t0.value, jnp.zeros(shape=batch_dims))
      self.assertAllEqual(progress_t0.valid, expected_valid)
    with self.subTest('end_position'):
      state_t90 = datatypes.update_state_by_log(state_t0, 90)
      progress_t90 = route.ProgressionMetric().compute(state_t90)
      self.assertAllEqual(progress_t90.value, jnp.ones(shape=batch_dims))
      self.assertAllEqual(progress_t90.valid, expected_valid)
    with self.subTest('half_way'):
      state_t50 = datatypes.update_state_by_log(state_t0, 50)
      progress_t50 = route.ProgressionMetric().compute(state_t50)
      self.assertAllLess(progress_t50.value, 1.0)
      self.assertAllGreater(progress_t50.value, 0.0)
      self.assertAllEqual(progress_t50.valid, expected_valid)


class OffRouteMetricTest(tf.test.TestCase):

  def test_raises_error_if_no_sdc_paths(self):
    traj = test_utils.create_test_trajectory_from_position(
        position=[0.0, 0.0, 0.0]
    )
    sim_state = create_sim_state_from_trajectories_and_paths(
        sim_trajectory=traj, log_trajectory=traj, sdc_paths=None
    )
    with self.assertRaises(ValueError):
      route.OffRouteMetric().compute(sim_state)

  def test_returns_invalid_if_sdc_traj_is_invalid(self):
    traj = test_utils.create_test_trajectory_from_position(
        position=[0.0, 0.0, 0.0]
    )
    traj = traj.replace(valid=jnp.zeros_like(traj.valid, dtype=jnp.bool_))
    # Create one valid off-route path.
    num_paths = 1
    num_points_per_path = 1
    paths_x = jnp.ones((num_paths, num_points_per_path))
    valid = jnp.ones((num_paths, num_points_per_path), dtype=jnp.bool_)
    on_route = jnp.zeros((num_paths, 1), dtype=jnp.bool_)
    paths = create_paths(paths_x, paths_x, paths_x, valid, on_route)
    sim_state = create_sim_state_from_trajectories_and_paths(
        sim_trajectory=traj, log_trajectory=traj, sdc_paths=paths
    )
    result = route.OffRouteMetric().compute(sim_state)
    # The metric should be invalid because the SDC trajectory is invalid.
    self.assertEqual(result.value, 0.0)
    self.assertFalse(result.valid)

  def test_returns_invalid_if_no_valid_paths(self):
    traj = test_utils.create_test_trajectory_from_position(
        position=[0.0, 0.0, 0.0]
    )
    # Create one invalid off-route path.
    num_paths = 1
    num_points_per_path = 1
    paths_x = jnp.ones((num_paths, num_points_per_path))
    valid = jnp.zeros((num_paths, num_points_per_path), dtype=jnp.bool_)
    on_route = jnp.zeros((num_paths, 1), dtype=jnp.bool_)
    paths = create_paths(paths_x, paths_x, paths_x, valid, on_route)
    sim_state = create_sim_state_from_trajectories_and_paths(
        sim_trajectory=traj, log_trajectory=traj, sdc_paths=paths
    )
    result = route.OffRouteMetric().compute(sim_state)
    # The metric should be invalid because the only path is invalid.
    self.assertEqual(result.value, 0.0)
    self.assertFalse(result.valid)

  def test_returns_zero_if_near_on_route_path(self):
    traj = test_utils.create_test_trajectory_from_position(
        position=[0.0, 0.0, 0.0]
    )
    num_paths = 1
    num_points_per_path = 3
    paths_x = jnp.ones((num_paths, num_points_per_path), dtype=jnp.float32)
    paths_y = jnp.zeros_like(paths_x)
    paths_z = jnp.zeros_like(paths_x)
    valid = jnp.ones((num_paths, num_points_per_path), dtype=jnp.bool_)
    on_route = jnp.ones((num_paths, 1), dtype=jnp.bool_)
    paths = create_paths(paths_x, paths_y, paths_z, valid, on_route)
    sim_state = create_sim_state_from_trajectories_and_paths(
        sim_trajectory=traj, log_trajectory=traj, sdc_paths=paths
    )
    result = route.OffRouteMetric().compute(sim_state)
    self.assertEqual(result.value, 0.0)
    self.assertTrue(result.valid)

  def test_returns_zero_for_test_data(self):
    config = _config.DatasetConfig(
        path=ROUTE_DATA_PATH,
        data_format=_config.DataFormat.TFRECORD,
        include_sdc_paths=True,
        max_num_rg_points=30000,
        batch_dims=(),
        num_paths=ROUTE_NUM_PATHS,
        num_points_per_path=ROUTE_NUM_POINTS_PER_PATH,
    )
    data_iter = dataloader.simulator_state_generator(config=config)
    state_t0 = next(data_iter)
    result = route.OffRouteMetric().compute(state_t0)
    self.assertEqual(result.value, 0.0)
    self.assertTrue(result.valid)

  def test_returns_dist_to_on_route_path_if_far_from_on_route_path(self):
    traj = test_utils.create_test_trajectory_from_position(
        position=[0.0, 0.0, 0.0]
    )
    # Make the first on-route path invalid, the second on-route path far enough
    # from the SDC state to be considered off-route, and the third on-route path
    # even farther from the SDC state than the second path.
    num_paths = 3
    num_points_per_path = 1
    path1_x = jnp.zeros(num_points_per_path, dtype=jnp.float32)
    closest_valid_path_x = route.OffRouteMetric.MAX_DISTANCE_TO_ROUTE_PATH + 1
    path2_x = jnp.full(
        num_points_per_path, closest_valid_path_x, dtype=jnp.float32
    )
    path3_x = jnp.full(
        num_points_per_path, closest_valid_path_x + 1, dtype=jnp.float32
    )
    paths_x = jnp.stack([path1_x, path2_x, path3_x], axis=0)
    paths_y = jnp.zeros_like(paths_x)
    paths_z = jnp.zeros_like(paths_x)
    valid = jnp.array([False, True, True], dtype=jnp.bool_)
    valid = jnp.broadcast_to(
        jnp.expand_dims(valid, axis=-1), (num_paths, num_points_per_path)
    )
    on_route = jnp.ones((num_paths, 1), dtype=jnp.bool_)
    paths = create_paths(paths_x, paths_y, paths_z, valid, on_route)
    sim_state = create_sim_state_from_trajectories_and_paths(
        sim_trajectory=traj, log_trajectory=traj, sdc_paths=paths
    )
    result = route.OffRouteMetric().compute(sim_state)
    self.assertEqual(result.value, closest_valid_path_x)
    self.assertTrue(result.valid)

  def test_returns_dist_to_on_route_path_if_close_enough_to_off_route_path(
      self,
  ):
    traj = test_utils.create_test_trajectory_from_position(
        position=[0.0, 0.0, 0.0]
    )
    # Make an on-route path that is farther from the SDC state than the
    # off-route path by more than MAX_RELATIVE_DISTANCE_TO_OFF_ROUTE_PATH.
    num_paths = 2
    num_points_per_path = 1
    dist_to_on_route_path = (
        route.OffRouteMetric.MAX_RELATIVE_DISTANCE_TO_OFF_ROUTE_PATH + 0.1
    )
    on_route_path_x = jnp.full(
        num_points_per_path, dist_to_on_route_path, dtype=jnp.float32
    )
    off_route_path_x = jnp.zeros(num_points_per_path, dtype=jnp.float32)
    paths_x = jnp.stack([on_route_path_x, off_route_path_x], axis=0)
    paths_y = jnp.zeros_like(paths_x)
    paths_z = jnp.zeros_like(paths_x)
    valid = jnp.ones((num_paths, num_points_per_path), dtype=jnp.bool_)
    on_route = jnp.array([[True], [False]], dtype=jnp.bool_)
    paths = create_paths(paths_x, paths_y, paths_z, valid, on_route)
    sim_state = create_sim_state_from_trajectories_and_paths(
        sim_trajectory=traj, log_trajectory=traj, sdc_paths=paths
    )
    result = route.OffRouteMetric().compute(sim_state)
    self.assertEqual(result.value, dist_to_on_route_path)
    self.assertTrue(result.valid)

  def test_returns_invalid_if_no_on_route_paths(self):
    traj = test_utils.create_test_trajectory_from_position(
        position=[0.0, 0.0, 0.0]
    )
    num_paths = 2
    num_points_per_path = 3
    paths_x = jnp.full((num_paths, num_points_per_path), 3, dtype=jnp.float32)
    paths_y = jnp.zeros_like(paths_x)
    paths_z = jnp.zeros_like(paths_x)
    valid = jnp.ones((num_paths, num_points_per_path), dtype=jnp.bool_)
    on_route = jnp.broadcast_to(
        jnp.array([[False]], dtype=jnp.bool_), (num_paths, 1)
    )
    paths = create_paths(paths_x, paths_y, paths_z, valid, on_route)
    sim_state = create_sim_state_from_trajectories_and_paths(
        sim_trajectory=traj, log_trajectory=traj, sdc_paths=paths
    )
    result = route.OffRouteMetric().compute(sim_state)
    self.assertEqual(result.value, 0.0)
    self.assertFalse(result.valid)

  def test_returns_invalid_if_no_valid_on_route_paths(self):
    traj = test_utils.create_test_trajectory_from_position(
        position=[0.0, 0.0, 0.0]
    )
    # Create an invalid on-route path far from the SDC position.
    paths_x = jnp.full((1, 1), 100, dtype=jnp.float32)
    valid = jnp.zeros_like(paths_x, dtype=jnp.bool_)
    on_route = jnp.ones_like(paths_x, dtype=jnp.bool_)
    paths = create_paths(paths_x, paths_x, paths_x, valid, on_route)
    sim_state = create_sim_state_from_trajectories_and_paths(
        sim_trajectory=traj, log_trajectory=traj, sdc_paths=paths
    )
    result = route.OffRouteMetric().compute(sim_state)
    self.assertEqual(result.value, 0.0)
    self.assertFalse(result.valid)

  def test_uses_sdc_state_at_timestep(self):
    # Create a trajectory where the first state is off-route and the second
    # state is on-route.
    traj_xyz_t0 = jnp.full((1, 3), 100, dtype=jnp.float32)
    traj_xyz_t1 = jnp.ones_like(traj_xyz_t0)
    # (num_objects=1, num_timesteps=2, 3)
    traj_xyz = jnp.stack([traj_xyz_t0, traj_xyz_t1], axis=-2)
    traj = create_test_trajectory_from_positions(traj_xyz)

    # Create an on-route path at (x = 0, y = 0).
    paths_x = jnp.zeros((1, 1))
    valid = jnp.ones_like(paths_x, dtype=jnp.bool_)
    on_route = jnp.ones_like(paths_x, dtype=jnp.bool_)
    paths = create_paths(paths_x, paths_x, paths_x, valid, on_route)
    sim_state = create_sim_state_from_trajectories_and_paths(
        sim_trajectory=traj, log_trajectory=traj, sdc_paths=paths
    )
    sim_state = sim_state.replace(timestep=jnp.ones((), dtype=jnp.int32))
    result = route.OffRouteMetric().compute(sim_state)
    # The metric should return zero because the second state is on-route.
    self.assertEqual(result.value, 0.0)
    self.assertTrue(result.valid)

  def test_uses_sim_trajectory(self):
    sim_traj = test_utils.create_test_trajectory_from_position(
        position=[0.0, 0.0, 0.0]
    )
    log_traj = test_utils.create_test_trajectory_from_position(
        position=[100.0, 0.0, 0.0]
    )
    num_paths = 1
    num_points_per_path = 3
    paths_x = jnp.zeros((num_paths, num_points_per_path), dtype=jnp.float32)
    valid = jnp.ones((num_paths, num_points_per_path), dtype=jnp.bool_)
    on_route = jnp.ones((num_paths, 1), dtype=jnp.bool_)
    paths = create_paths(paths_x, paths_x, paths_x, valid, on_route)
    sim_state = create_sim_state_from_trajectories_and_paths(
        sim_trajectory=sim_traj, log_trajectory=log_traj, sdc_paths=paths
    )
    result = route.OffRouteMetric().compute(sim_state)
    self.assertEqual(result.value, 0.0)
    self.assertTrue(result.valid)


def create_paths(
    x: jax.Array,
    y: jax.Array,
    z: jax.Array,
    valid: jax.Array,
    on_route: jax.Array,
    arc_length: Optional[jax.Array] = None,
) -> datatypes.Paths:
  ids = jnp.zeros_like(x, dtype=jnp.int32)
  arc_length = (
      arc_length
      if arc_length is not None
      else jnp.zeros_like(x, dtype=jnp.float32)
  )
  paths = datatypes.Paths(
      x=x,
      y=y,
      z=z,
      ids=ids,
      valid=valid,
      arc_length=arc_length,
      on_route=on_route,
  )
  paths.validate()
  return paths


def create_sim_state_from_trajectories_and_paths(
    sim_trajectory: datatypes.Trajectory,
    log_trajectory: datatypes.Trajectory,
    sdc_paths: Optional[datatypes.Paths],
) -> datatypes.SimulatorState:
  roadgraph_points = test_utils.create_test_map_element(
      element_type=datatypes.MapElementIds.ROAD_EDGE_BOUNDARY,
      position=[0.0, 0.0, 0.0],
      direction=[0.0, 1.0, 0.0],
      repeat=10,
  )
  tl = test_utils.create_invalid_traffic_lights()
  return datatypes.SimulatorState(
      roadgraph_points=roadgraph_points,
      sim_trajectory=sim_trajectory,
      log_trajectory=log_trajectory,
      log_traffic_light=tl,
      object_metadata=test_utils.create_metadata(sim_trajectory.num_objects),
      timestep=jnp.zeros((), dtype=jnp.int32),
      sdc_paths=sdc_paths,
  )


def create_test_trajectory_from_positions(
    xyz: jax.Array,
) -> datatypes.Trajectory:
  if len(xyz.shape) < 3:
    raise ValueError('xyz must have at least 3 dimensions.')
  prefix_shape = xyz.shape[:-1]
  vel_x = jnp.zeros(prefix_shape, dtype=jnp.float32)
  vel_y = jnp.zeros(prefix_shape, dtype=jnp.float32)
  yaw = jnp.zeros(prefix_shape, dtype=jnp.float32)
  valid = jnp.ones(prefix_shape, dtype=jnp.bool_)
  lwh = jnp.ones(prefix_shape, jnp.float32)
  timestamp_micros = jnp.zeros(prefix_shape, dtype=jnp.int32)
  traj = datatypes.Trajectory(
      x=xyz[..., 0],
      y=xyz[..., 1],
      z=xyz[..., 2],
      vel_x=vel_x,
      vel_y=vel_y,
      yaw=yaw,
      valid=valid,
      timestamp_micros=timestamp_micros,
      width=lwh,
      length=lwh,
      height=lwh,
  )
  traj.validate()
  return traj


if __name__ == '__main__':
  tf.test.main()
