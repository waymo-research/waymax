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

import functools

import jax
import jax.numpy as jnp
import tensorflow as tf

from absl.testing import parameterized
from waymax import config as _config
from waymax.dataloader import womd_dataloader
from waymax.datatypes import observation
from waymax.datatypes import operations
from waymax.datatypes import roadgraph
from waymax.datatypes import simulator_state
from waymax.utils import test_utils

TEST_DATA_PATH = test_utils.ROUTE_DATA_PATH


class RoadgraphTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.rg = roadgraph.RoadgraphPoints(
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
    self.rg.validate()

  def test_top_k_roadgraph_returns_correct_output_fewer_points(self):
    xyz_and_direction = jnp.array(
        [
            [2.0, 2.0, 2.0, 0.0, 0.0],
            [3.0, 3.0, 3.0, 2.0, 2.0],
            [2.7, 2.7, 3.0, 3.0, 3.0],
        ],
        dtype=jnp.float32,
    )

    types_ids = jnp.array([[1, 0], [3, 1], [0, 0]], dtype=jnp.int32)

    valid = jnp.array([True, True, False])

    roadgraph_single = roadgraph.RoadgraphPoints(
        x=xyz_and_direction[..., 0],
        y=xyz_and_direction[..., 1],
        z=xyz_and_direction[..., 2],
        dir_x=xyz_and_direction[..., 3],
        dir_y=xyz_and_direction[..., 4],
        dir_z=xyz_and_direction[..., 5],
        types=types_ids[..., 0],
        ids=types_ids[..., 1],
        valid=valid,
    )
    # Creates two copies of road graph data for testing.
    rg = jax.tree_util.tree_map(
        lambda x: jnp.repeat(x[jnp.newaxis, :], repeats=2, axis=0),
        roadgraph_single,
    )
    reference = jnp.array([[2.1, 2.1], [2.7, 2.7]])

    top_k_roadgraph = roadgraph.filter_topk_roadgraph_points(
        rg, reference, topk=1
    )
    self.assertAllClose(top_k_roadgraph.x, jnp.array([[2.0], [3.0]]))
    self.assertAllClose(
        top_k_roadgraph.types, jnp.array([[1], [3]], dtype=jnp.int32)
    )
    self.assertAllClose(
        top_k_roadgraph.valid, jnp.array([[True], [True]], dtype=jnp.int32)
    )

  @parameterized.parameters(((),), ((2, 1),))
  def test_filter_roadgraph_points_with_jit_produces_same_results(
      self, batch_dims
  ):
    config = _config.DatasetConfig(
        path=TEST_DATA_PATH,
        max_num_objects=8,
        batch_dims=batch_dims,
        data_format=_config.DataFormat.TFRECORD,
    )
    state_t0 = next(womd_dataloader.simulator_state_generator(config))
    state_t10 = simulator_state.update_state_by_log(state_t0, 10)

    global_obs = observation.global_observation_from_state(
        state_t10,
        obs_num_steps=5,
        num_obj=config.max_num_objects,
    )
    # Shape: (..., num_agents, num_timesteps=1, 2)
    reference_xy = operations.dynamic_slice(
        state_t10.sim_trajectory.xy, state_t10.timestep, 1, axis=-2
    )
    # Shape: (..., num_agents, 2)
    reference_xy = reference_xy[..., 0, :]
    filter_fn = functools.partial(
        roadgraph.filter_topk_roadgraph_points,
        reference_points=reference_xy,
        topk=2000,
    )
    roadgraph_topk = filter_fn(global_obs.roadgraph_static_points)
    jit_roadgraph_topk = jax.jit(filter_fn)(global_obs.roadgraph_static_points)
    self.assertEqual(roadgraph_topk, jit_roadgraph_topk)

  def test_roadgraph_structure_shapes_are_reported_correctly(self):
    rg = roadgraph.RoadgraphPoints(
        x=jnp.zeros((10, 100)),
        y=jnp.zeros((10, 100)),
        z=jnp.zeros((10, 100)),
        dir_x=jnp.zeros((10, 100)),
        dir_y=jnp.zeros((10, 100)),
        dir_z=jnp.zeros((10, 100)),
        types=jnp.zeros((10, 100)),
        ids=jnp.zeros((10, 100)),
        valid=jnp.zeros((10, 100)),
    )
    self.assertAllEqual(rg.shape, (10, 100))
    self.assertEqual(rg.num_points, 100)

  def test_roadgraph_xy_returns_correct_values(self):
    self.assertAllClose(self.rg.xy, jnp.array([[0, 1]]))

  def test_roadgraph_xyz_returns_correct_values(self):
    self.assertAllClose(self.rg.xyz, jnp.array([[0, 1, 2]]))

  def test_roadgraph_dir_xy_returns_correct_values(self):
    self.assertAllClose(self.rg.dir_xy, jnp.array([[3, 4]]))

  def test_roadgraph_dir_xyz_returns_correct_values(self):
    self.assertAllClose(self.rg.dir_xyz, jnp.array([[3, 4, 5]]))

  def test_roadgraph_equality_returns_correctly(self):
    with self.subTest('Equality=True'):
      self.assertEqual(self.rg, self.rg)
    with self.subTest('Equality=False'):
      self.assertNotEqual(self.rg, self.rg.replace(x=jnp.array([1])))

  def test_roadgraph_validate_asserts_if_improperly_created(self):
    with self.subTest('IdsWrongType'):
      with self.assertRaises(AssertionError):
        self.rg.replace(x=jnp.zeros((1), dtype=jnp.int32)).validate()
    with self.subTest('ObjectTypesWrongType'):
      with self.assertRaises(AssertionError):
        self.rg.replace(y=jnp.zeros((1), dtype=jnp.int32)).validate()
    with self.subTest('ObjectTypesWrongType'):
      with self.assertRaises(AssertionError):
        self.rg.replace(z=jnp.zeros((1), dtype=jnp.int32)).validate()
    with self.subTest('ObjectTypesWrongType'):
      with self.assertRaises(AssertionError):
        self.rg.replace(dir_x=jnp.zeros((1), dtype=jnp.int32)).validate()
    with self.subTest('ObjectTypesWrongType'):
      with self.assertRaises(AssertionError):
        self.rg.replace(dir_y=jnp.zeros((1), dtype=jnp.int32)).validate()
    with self.subTest('ObjectTypesWrongType'):
      with self.assertRaises(AssertionError):
        self.rg.replace(dir_z=jnp.zeros((1), dtype=jnp.int32)).validate()
    with self.subTest('ObjectTypesWrongType'):
      with self.assertRaises(AssertionError):
        self.rg.replace(types=jnp.zeros((1), dtype=jnp.float32)).validate()
    with self.subTest('ObjectTypesWrongType'):
      with self.assertRaises(AssertionError):
        self.rg.replace(ids=jnp.zeros((1), dtype=jnp.float32)).validate()
    with self.subTest('ObjectTypesWrongType'):
      with self.assertRaises(AssertionError):
        self.rg.replace(valid=jnp.zeros((1), dtype=jnp.float32)).validate()

    with self.subTest('ShapesNotTheSame'):
      error = (
          '[Chex] Assertion assert_equal_shape failed: Arrays have '
          'different shapes: [(1,), (1,), (1,), (1,), (1,), (1,), (1,), '
          '(1,), (1, 1)].'
      )
      with self.assertRaisesWithLiteralMatch(AssertionError, error):
        self.rg.replace(valid=jnp.zeros((1, 1), dtype=jnp.bool_)).validate()


if __name__ == '__main__':
  tf.test.main()
