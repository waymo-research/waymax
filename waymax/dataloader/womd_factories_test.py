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

import jax
from jax import numpy as jnp
import tensorflow as tf

from absl.testing import parameterized
from waymax.dataloader import womd_factories
from waymax.datatypes import roadgraph
from waymax.datatypes import route
from waymax.datatypes import traffic_lights
from waymax.utils import test_utils


class DataStructTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters(((),), ((2, 1),))
  def test_metadata_from_dict_consistent_with_real_data(self, batch_dims):
    dataset = test_utils.make_test_dataset(batch_dims=batch_dims)
    data_dict = next(dataset.as_numpy_iterator())
    obj_metadata = womd_factories.object_metadata_from_womd_dict(data_dict)
    self.assertEqual(obj_metadata.shape, batch_dims + (128,))

  @parameterized.parameters(((),), ((2, 1),))
  def test_roadgraph_from_dict_consistent_with_real_data(self, batch_dims):
    dataset = test_utils.make_test_dataset(batch_dims=batch_dims)
    data_dict = next(dataset.as_numpy_iterator())
    rg_pts = womd_factories.roadgraph_from_womd_dict(data_dict)

    num_points = 30000
    expected_shape = batch_dims + (num_points,)
    self.assertEqual(rg_pts.shape, expected_shape)
    self.assertEqual(rg_pts.num_points, num_points)

  def test_roadgraph_from_womd_dict_returns_correct_values(self):
    rg = roadgraph.RoadgraphPoints(
        x=jnp.array([0], dtype=jnp.float32),
        y=jnp.array([1], dtype=jnp.float32),
        z=jnp.array([2], dtype=jnp.float32),
        dir_x=jnp.array([3], dtype=jnp.float32),
        dir_y=jnp.array([4], dtype=jnp.float32),
        dir_z=jnp.array([5], dtype=jnp.float32),
        types=jnp.array([6], dtype=jnp.int32),
        ids=jnp.array([7], dtype=jnp.int32),
        valid=jnp.array([True], dtype=jnp.bool_),
    )
    rg.validate()
    prefix = 'roadgraph_samples'
    example = {
        f'{prefix}/xyz': jnp.array([[0, 1, 2]]),
        f'{prefix}/dir': jnp.array([[3, 4, 5]]),
        f'{prefix}/type': jnp.array([[6]]),
        f'{prefix}/id': jnp.array([[7]]),
        f'{prefix}/valid': jnp.array([[True]]),
    }
    self.assertEqual(
        rg, womd_factories.roadgraph_from_womd_dict(example, prefix)
    )

  @parameterized.parameters(((),), ((2, 1),))
  def test_sim_state_from_dict_consistent_with_real_data(self, batch_dims):
    dataset = test_utils.make_test_dataset(batch_dims=batch_dims)
    data_dict = next(dataset.as_numpy_iterator())
    sim_state = womd_factories.simulator_state_from_womd_dict(
        data_dict, time_key='all'
    )
    self.assertEqual(sim_state.shape, batch_dims)

    flat_sim_traj, _ = jax.tree_util.tree_flatten(sim_state.sim_trajectory)
    flat_log_traj, _ = jax.tree_util.tree_flatten(sim_state.log_trajectory)
    with self.subTest('FirstCopied'):
      for x, y in zip(flat_sim_traj, flat_log_traj):
        self.assertAllClose(x[..., 0], y[..., 0])
    with self.subTest('RestAreZeros'):
      for x in flat_sim_traj:
        self.assertAllClose(
            x[..., 1:], jnp.zeros_like(x[..., 1:], dtype=x.dtype)
        )

  @parameterized.parameters(((),), ((2, 1),))
  def test_traj_from_dict_consistent_with_real_data(self, batch_dims):
    dataset = test_utils.make_test_dataset(
        batch_dims=batch_dims, aggregate_timesteps=False
    )
    data_dict = next(dataset.as_numpy_iterator())
    traj = womd_factories.trajectory_from_womd_dict(data_dict, time_key='past')

    num_obj, num_timesteps = 128, 10
    expected_shape = batch_dims + (num_obj, num_timesteps)
    self.assertEqual(traj.shape, expected_shape)
    self.assertEqual(traj.num_objects, num_obj)
    self.assertEqual(traj.num_timesteps, num_timesteps)
    self.assertTrue(_check_values_are_all_same(traj.length, axis=-1))
    self.assertTrue(_check_values_are_all_same(traj.width, axis=-1))
    self.assertTrue(_check_values_are_all_same(traj.height, axis=-1))

  def test_from_womd_dict_works_for_trajectory(self):
    example = {
        'state/all/x': jnp.array([0]),
        'state/all/y': jnp.array([1]),
        'state/all/z': jnp.array([2]),
        'state/all/velocity_x': jnp.array([3]),
        'state/all/velocity_y': jnp.array([4]),
        'state/all/bbox_yaw': jnp.array([5]),
        'state/all/valid': jnp.array([True]),
        'state/all/length': jnp.array([7]),
        'state/all/width': jnp.array([8]),
        'state/all/height': jnp.array([9]),
        'state/all/timestamp_micros': jnp.array([10]),
    }
    traj = womd_factories.trajectory_from_womd_dict(example)
    # Must be ok after validation.
    traj.validate()
    self.assertAllClose(traj.x, example['state/all/x'])
    self.assertAllClose(traj.y, example['state/all/y'])
    self.assertAllClose(traj.z, example['state/all/z'])
    self.assertAllClose(traj.vel_x, example['state/all/velocity_x'])
    self.assertAllClose(traj.vel_y, example['state/all/velocity_y'])
    self.assertAllClose(traj.yaw, example['state/all/bbox_yaw'])
    self.assertAllClose(traj.valid, example['state/all/valid'])
    self.assertAllClose(traj.length, example['state/all/length'])
    self.assertAllClose(traj.width, example['state/all/width'])
    self.assertAllClose(traj.height, example['state/all/height'])
    self.assertAllClose(
        traj.timestamp_micros, example['state/all/timestamp_micros']
    )

  @parameterized.parameters(((),), ((2, 1),))
  def test_tls_from_dict_consistent_with_real_data(self, batch_dims):
    dataset = test_utils.make_test_dataset(
        batch_dims=batch_dims, aggregate_timesteps=False
    )

    data_dict = next(dataset.as_numpy_iterator())
    tls = womd_factories.traffic_lights_from_womd_dict(
        data_dict, time_key='past'
    )
    self.assertEqual(tls.shape, batch_dims + (16, 10))
    self.assertEqual(tls.num_traffic_lights, 16)
    self.assertEqual(tls.num_timesteps, 10)

  def test_traffic_lights_from_womd_dict_works_properly(self):
    example = {
        'traffic_light_state/all/x': jnp.zeros((91, 16), dtype=jnp.float32),
        'traffic_light_state/all/y': jnp.ones((91, 16)) * 1.0,
        'traffic_light_state/all/z': jnp.ones((91, 16)) * 2.0,
        'traffic_light_state/all/state': jnp.ones((91, 16)) * 3,
        'traffic_light_state/all/id': jnp.ones((91, 16)) * 4,
        'traffic_light_state/all/valid': jnp.ones((91, 16), dtype=jnp.bool_),
    }
    tls = womd_factories.traffic_lights_from_womd_dict(example)
    expected_tls = traffic_lights.TrafficLights(
        x=jnp.zeros((16, 91), dtype=jnp.float32),
        y=jnp.ones((16, 91)) * 1.0,
        z=jnp.ones((16, 91)) * 2.0,
        state=jnp.ones((16, 91)) * 3,
        lane_ids=jnp.ones((16, 91)) * 4,
        valid=jnp.ones((16, 91), dtype=jnp.bool_),
    )
    self.assertEqual(tls, expected_tls)

  def test_traffic_lights_from_womd_dict_raises_if_improper_key_provided(self):
    with self.assertRaisesWithLiteralMatch(
        ValueError, 'time_key time is not supported.'
    ):
      womd_factories.traffic_lights_from_womd_dict({}, time_key='time')

  @parameterized.parameters(((),), ((2, 1),))
  def test_route_paths_from_dict_consistent_with_real_data(self, batch_dims):
    num_paths = 30
    num_points_per_path = 200
    dataset = test_utils.make_test_dataset(
        batch_dims=batch_dims,
        aggregate_timesteps=True,
        include_sdc_paths=True,
        num_paths=num_paths,
        num_points_per_path=num_points_per_path,
    )
    data_dict = next(dataset.as_numpy_iterator())
    routes = womd_factories.paths_from_womd_dict(data_dict)
    self.assertEqual(
        routes.shape, batch_dims + (num_paths, num_points_per_path)
    )
    self.assertEqual(routes.num_points_per_path, num_points_per_path)
    self.assertEqual(routes.num_paths, num_paths)

  def test_form_womd_dict_returns_correct_routes(self):
    example = {
        'path_samples/xyz': jnp.array([[0.0, 1.0, 2.0]]),
        'path_samples/id': jnp.array([3]),
        'path_samples/valid': jnp.array([True]),
        'path_samples/arc_length': jnp.array([4.0]),
        'path_samples/on_route': jnp.array([True]),
    }
    exp_routes = route.Paths(
        x=jnp.array([0.0]),
        y=jnp.array([1.0]),
        z=jnp.array([2.0]),
        ids=jnp.array([3]),
        valid=jnp.array([True]),
        arc_length=jnp.array([4.0]),
        on_route=jnp.array([True]),
    )
    self.assertAllClose(
        exp_routes, womd_factories.paths_from_womd_dict(example)
    )


def _check_values_are_all_same(value: jax.Array, axis: int = 0) -> jax.Array:
  """Checks to make sure all of the values reduced along axis are equal."""
  array_slice = [slice(dim) for dim in list(value.shape)]
  array_slice[axis] = 0
  tile_values = [1] * len(value.shape)
  tile_values[axis] = value.shape[axis]
  one_value_along_axis = jnp.tile(
      jnp.expand_dims(value[tuple(array_slice)], axis=axis), tile_values
  )
  return jnp.all(jnp.equal(one_value_along_axis, value))


if __name__ == '__main__':
  tf.test.main()
