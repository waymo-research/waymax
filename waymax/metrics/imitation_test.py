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

"""Tests for Waymax imitation metrics."""

from jax import numpy as jnp
import tensorflow as tf

from absl.testing import parameterized
from waymax import dataloader
from waymax import datatypes
from waymax.metrics import imitation
from waymax.utils import test_utils


# TODO(b/267827375): test that the metric result validity mask is correct.
class LogDivergenceMetricTest(tf.test.TestCase, parameterized.TestCase):

  def test_metric_runs_from_real_data(self):
    dataset = test_utils.make_test_dataset()
    data_dict = next(dataset.as_numpy_iterator())
    sim_state_init = dataloader.simulator_state_from_womd_dict(
        data_dict, time_key='all'
    )
    result = imitation.LogDivergenceMetric().compute(sim_state_init)
    self.assertEqual(result.value.shape, (128,))
    self.assertEqual(result.valid.shape, (128,))

  @parameterized.parameters(((1,),), ((3, 5),), ((6, 8, 9),))
  def test_metric_returns_correct_results(self, dimensions):
    object_state = datatypes.Trajectory.zeros(dimensions).replace(
        x=jnp.ones(dimensions),
        y=jnp.ones(dimensions),
    )
    log_state = object_state.replace(
        x=jnp.ones(dimensions) * 3.0, y=jnp.ones(dimensions) * 1.0
    )
    result = imitation.LogDivergenceMetric().compute_log_divergence(
        object_state.xy, log_state.xy
    )
    expected = jnp.ones(dimensions) * 2.0
    self.assertAllClose(result, expected)


if __name__ == '__main__':
  tf.test.main()
