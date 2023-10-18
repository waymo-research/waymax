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

"""Tests for Waymax roadgraph metrics."""

from jax import numpy as jnp
import numpy as np
import tensorflow as tf

from absl.testing import parameterized
from waymax import config as _config
from waymax import dataloader
from waymax import datatypes
from waymax.metrics import roadgraph
from waymax.utils import test_utils

ROUTE_DATA_PATH = test_utils.ROUTE_DATA_PATH
ROUTE_NUM_PATHS = test_utils.ROUTE_NUM_PATHS
ROUTE_NUM_POINTS_PER_PATH = test_utils.ROUTE_NUM_POINTS_PER_PATH


class OffroadMetricTest(tf.test.TestCase, parameterized.TestCase):

  def test_offroad_metric_runs_from_real_data(self):
    dataset = test_utils.make_test_dataset()
    data_dict = next(dataset.as_numpy_iterator())
    sim_state_init = dataloader.simulator_state_from_womd_dict(
        data_dict, time_key='all'
    )
    result = roadgraph.OffroadMetric().compute(sim_state_init)
    self.assertEqual(result.value.shape, (128,))
    self.assertEqual(result.valid.shape, (128,))

  def test_offroad_metric_runs_from_sim_data_with_offroad(self):
    sim_state = test_utils.simulator_state_with_offroad()
    result = roadgraph.OffroadMetric().compute(sim_state)
    self.assertAllEqual(result.value, (1.0,))
    self.assertAllEqual(result.valid, (True,))

  def test_offroad_metric_runs_from_sim_data_without_offroad(self):
    sim_state = test_utils.simulator_state_without_offroad()
    result = roadgraph.OffroadMetric().compute(sim_state)
    self.assertAllEqual(result.value, (0.0,))
    self.assertAllEqual(result.valid, (True,))


class WrongWayMetricTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    batch_dims = (1, 2)
    self.config = _config.DatasetConfig(
        path=ROUTE_DATA_PATH,
        data_format=_config.DataFormat.TFRECORD,
        include_sdc_paths=True,
        max_num_rg_points=30000,
        batch_dims=batch_dims,
        num_paths=ROUTE_NUM_PATHS,
        num_points_per_path=ROUTE_NUM_POINTS_PER_PATH,
    )
    data_iter = dataloader.simulator_state_generator(config=self.config)
    self.state_t0 = next(data_iter)
    self.state_t50 = datatypes.update_state_by_log(self.state_t0, 50)

  def test_gt_data_is_on_route(self):
    out_t0 = roadgraph.WrongWayMetric().compute(self.state_t0)
    out_t50 = roadgraph.WrongWayMetric().compute(self.state_t50)
    expected_value = jnp.zeros(shape=self.config.batch_dims)
    expected_valid = jnp.ones_like(expected_value, dtype=jnp.bool_)
    self.assertAllEqual(out_t0.value, expected_value)
    self.assertAllEqual(out_t0.valid, expected_valid)
    self.assertAllEqual(out_t50.value, expected_value)
    self.assertAllEqual(out_t50.valid, expected_valid)

  def test_sample_position_is_off_route(self):
    # In both case, obj 3's position is considered off-route for SDC.
    num_obj = self.state_t50.sim_trajectory.num_objects
    obj0_is_sdc = np.arange(num_obj) == 3
    for b in reversed(self.config.batch_dims):
      obj0_is_sdc = np.repeat(obj0_is_sdc[np.newaxis, ...], b, axis=0)
    metadata = self.state_t50.object_metadata.replace(is_sdc=obj0_is_sdc)
    state = self.state_t50.replace(object_metadata=metadata)
    out = roadgraph.WrongWayMetric().compute(state)
    self.assertAllClose(out.value, [[4.540114, 4.106123]])
    self.assertAllEqual(out.valid, [[True, True]])


class RoadgraphTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(('is_offroad', True), ('is_onroad', False))
  def test_check_offroad_is_true(self, is_offroad):
    if is_offroad:
      simulator_state = test_utils.simulator_state_with_offroad()
    else:
      simulator_state = test_utils.simulator_state_without_offroad()
    output = roadgraph.is_offroad(
        simulator_state.sim_trajectory, simulator_state.roadgraph_points
    )
    self.assertAllEqual(output, [is_offroad])

  def test_check_signed_distance_with_offroad(self):
    simulator_state = test_utils.simulator_state_with_offroad()
    signed_distance = (
        roadgraph.compute_signed_distance_to_nearest_road_edge_point(
            query_points=simulator_state.sim_trajectory.xyz[:, 0],
            roadgraph_points=simulator_state.roadgraph_points,
        )
    )
    self.assertAllClose(signed_distance, [4.2426405])

  def test_check_signed_distance_with_onroad(self):
    simulator_state = test_utils.simulator_state_without_offroad()
    signed_distance = (
        roadgraph.compute_signed_distance_to_nearest_road_edge_point(
            query_points=simulator_state.sim_trajectory.xyz[:, 0],
            roadgraph_points=simulator_state.roadgraph_points,
        )
    )
    self.assertAllClose(signed_distance, [-1.2165525])


if __name__ == '__main__':
  tf.test.main()
