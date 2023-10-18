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

"""Tests for womd_dataloader."""

import jax
import tensorflow as tf

from absl.testing import parameterized
from waymax import config as _config
from waymax.dataloader import womd_dataloader
from waymax.utils import test_utils


class WomdDataloaderTest(parameterized.TestCase, tf.test.TestCase):

  def test_example_state_is_consistent_with_real_data(self):
    config = _config.DatasetConfig(
        path=test_utils.ROUTE_DATA_PATH,
        max_num_objects=32,
        data_format=_config.DataFormat.TFRECORD,
    )
    dummy_state = test_utils.make_zeros_state(config=config)
    data_iter = womd_dataloader.simulator_state_generator(config=config)
    real_state = next(data_iter)
    flags = jax.tree_util.tree_map(
        lambda x, y: (x.shape == y.shape) & (x.dtype == y.dtype),
        dummy_state,
        real_state,
    )
    self.assertTrue(jax.tree_util.tree_all(flags))

  @parameterized.parameters(True, False)
  def test_simulator_state_generator_runs_end2end(self, distributed):
    config = _config.DatasetConfig(
        path=test_utils.ROUTE_DATA_PATH,
        max_num_objects=32,
        distributed=distributed,
        data_format=_config.DataFormat.TFRECORD,
    )
    data_iter = womd_dataloader.simulator_state_generator(config=config)
    example = next(data_iter)
    if distributed:
      exp_shape = (jax.local_device_count(),) + (32, 91, 2)
      example_value = example.log_trajectory.xy[0, 0, :3, 0]
    else:
      exp_shape = (32, 91, 2)
      example_value = example.log_trajectory.xy[0, :3, 0]
    self.assertEqual(example.log_trajectory.xy.shape, exp_shape)
    # Deterministic check on a sample value
    self.assertEqual(tuple(example_value), (1120.4044, 1119.4, 1118.4131))

  def test_raises_error_if_include_sdc_paths_without_dims(self):
    config = _config.DatasetConfig(
        path=test_utils.ROUTE_DATA_PATH,
        data_format=_config.DataFormat.TFRECORD,
        include_sdc_paths=True,
        batch_dims=(1, 2),
        num_paths=None,
        num_points_per_path=None,
    )

    with self.assertRaises(ValueError):
      next(womd_dataloader.simulator_state_generator(config=config))

  def test_route_in_sim_state(self):
    batch_dims = (1, 2)
    route_num_paths = 30
    route_num_points_per_path = 200
    config = _config.DatasetConfig(
        path=test_utils.ROUTE_DATA_PATH,
        data_format=_config.DataFormat.TFRECORD,
        include_sdc_paths=True,
        batch_dims=batch_dims,
        num_paths=route_num_paths,
        num_points_per_path=route_num_points_per_path,
    )
    data_iter = womd_dataloader.simulator_state_generator(config=config)
    sim_state = next(data_iter)
    self.assertEqual(
        sim_state.sdc_paths.shape,
        batch_dims + (route_num_paths, route_num_points_per_path),
    )
    self.assertEqual(
        tuple(sim_state.sdc_paths.x[0, 0, 0, :3]),
        (1210.6556, 1209.6846, 1208.7137),
    )


if __name__ == '__main__':
  tf.test.main()
