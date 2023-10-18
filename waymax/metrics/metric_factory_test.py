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

"""Tests the metric factory function."""

from jax import numpy as jnp
import tensorflow as tf

from absl.testing import parameterized
from waymax import config as _config
from waymax import dataloader
from waymax.metrics import metric_factory
from waymax.utils import test_utils

TEST_DATA_PATH = test_utils.ROUTE_DATA_PATH


class MetricFactoryTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters(((),), ((2, 1),))
  def test_all_false_flags_results_in_empty_results_dict(self, batch_dims):
    config = _config.EnvironmentConfig(
        metrics=_config.MetricsConfig(
            run_log_divergence=False, run_overlap=False, run_offroad=False
        )
    )

    dataset = test_utils.make_test_dataset(batch_dims=batch_dims)
    data_dict = next(dataset.as_numpy_iterator())
    sim_state = dataloader.simulator_state_from_womd_dict(
        data_dict, time_key='all'
    )

    metric_results = metric_factory.run_metrics(
        simulator_state=sim_state, metrics_config=config.metrics
    )

    self.assertEmpty(metric_results)

  @parameterized.parameters(((),), ((2, 1),))
  def test_true_flags_results_in_correct_number_of_results(self, batch_dims):
    config = _config.EnvironmentConfig(
        metrics=_config.MetricsConfig(
            run_log_divergence=True, run_overlap=True, run_offroad=True
        )
    )

    dataset = test_utils.make_test_dataset(batch_dims=batch_dims)
    data_dict = next(dataset.as_numpy_iterator())
    sim_state = dataloader.simulator_state_from_womd_dict(
        data_dict, time_key='all'
    )

    metric_results = metric_factory.run_metrics(
        simulator_state=sim_state, metrics_config=config.metrics
    )
    metric_names = ['log_divergence', 'overlap', 'offroad']

    with self.subTest('check_correct_number_of_elements'):
      self.assertLen(metric_results, 3)
    with self.subTest('check_correct_names'):
      self.assertSameElements(metric_results.keys(), metric_names)

    with self.subTest('check_correct_shapes'):
      self.assertAllEqual(
          metric_results['log_divergence'].shape,
          list(batch_dims) + [sim_state.num_objects],
      )

  def test_offroad_is_detected_in_metric(self):
    config = _config.EnvironmentConfig(
        metrics=_config.MetricsConfig(
            run_log_divergence=False, run_overlap=False, run_offroad=True
        )
    )

    sim_state = test_utils.simulator_state_with_offroad()
    metric_results = metric_factory.run_metrics(
        simulator_state=sim_state, metrics_config=config.metrics
    )
    offroad_metric_result = metric_results['offroad']

    with self.subTest('other_metrics_are_missing'):
      self.assertNotIn('overlap', metric_results.keys())
    with self.subTest('value'):
      self.assertAllEqual(tuple(offroad_metric_result.value), (1,))
    with self.subTest('valid'):
      self.assertAllEqual(tuple(offroad_metric_result.valid), (True,))

  @parameterized.named_parameters(
      (
          'valid',
          True,
          1.0,
      ),
      (
          'invalid',
          False,
          0.0,
      ),
  )
  def test_overlap_is_detected_in_metric(self, valid, expected):
    config = _config.EnvironmentConfig(
        metrics=_config.MetricsConfig(
            run_log_divergence=False, run_overlap=True, run_offroad=False
        )
    )

    sim_state = test_utils.simulator_state_with_overlap()
    sim_traj = sim_state.sim_trajectory
    sim_traj = sim_traj.replace(valid=jnp.full_like(sim_traj.valid, valid))
    sim_state = sim_state.replace(sim_trajectory=sim_traj)
    metric_results = metric_factory.run_metrics(
        simulator_state=sim_state, metrics_config=config.metrics
    )
    overlap_metric_result = metric_results['overlap']

    with self.subTest('value'):
      self.assertAllEqual(
          overlap_metric_result.value,
          jnp.full(sim_traj.num_objects, expected),
      )
    with self.subTest('valid'):
      self.assertAllEqual(
          overlap_metric_result.valid,
          jnp.full(sim_traj.num_objects, valid, dtype=jnp.bool_),
      )



if __name__ == '__main__':
  tf.test.main()
