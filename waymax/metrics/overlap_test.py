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

"""Tests for Waymax overlap metrics."""

import jax.numpy as jnp
import tensorflow as tf
from waymax import dataloader
from waymax.metrics import overlap
from waymax.utils import test_utils

from absl.testing import parameterized


class OverlapMetricTest(tf.test.TestCase, parameterized.TestCase):

  def test_metric_runs_from_real_data(self):
    dataset = test_utils.make_test_dataset()
    data_dict = next(dataset.as_numpy_iterator())
    sim_state_init = dataloader.simulator_state_from_womd_dict(
        data_dict, time_key='all'
    )
    result = overlap.OverlapMetric().compute(sim_state_init)
    self.assertEqual(result.value.shape, (128,))
    self.assertEqual(result.valid.shape, (128,))

  def test_metric_detects_zero_overlaps(self):
    traj_with_no_overlaps = test_utils.simulated_trajectory_no_overlap()
    metric = overlap.OverlapMetric().compute_overlap(traj_with_no_overlaps)
    num_objects = traj_with_no_overlaps.num_objects
    with self.subTest('value'):
      self.assertAllEqual(
          metric.value,
          jnp.zeros(
              num_objects,
          ),
      )
    with self.subTest('valid'):
      self.assertAllEqual(metric.valid, traj_with_no_overlaps.valid[..., 0])

  def test_metric_detects_all_overlaps(self):
    traj_with_overlap = test_utils.simulated_trajectory_with_overlap()
    metric = overlap.OverlapMetric().compute_overlap(traj_with_overlap)
    num_objects = traj_with_overlap.num_objects
    with self.subTest('value'):
      self.assertAllEqual(
          metric.value,
          jnp.ones(
              num_objects,
          ),
      )
    with self.subTest('valid'):
      self.assertAllEqual(metric.valid, traj_with_overlap.valid[..., 0])


if __name__ == '__main__':
  tf.test.main()
