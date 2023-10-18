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

"""Tests for metrics."""

from jax import numpy as jnp
import tensorflow as tf

from absl.testing import parameterized
from waymax import datatypes
from waymax.metrics import abstract_metric


class SimpleMetric(abstract_metric.AbstractMetric):

  def compute(
      self, simulator_state: datatypes.SimulatorState
  ) -> abstract_metric.MetricResult:
    value = jnp.ndarray([1, 2, 3])
    valid = jnp.ones_like(value, dtype=jnp.bool_)
    return abstract_metric.MetricResult.create_and_validate(value, valid)


class AbstractMetricTest(tf.test.TestCase):

  def test_abstract_metric_instantiates(self):
    metric = SimpleMetric()
    self.assertIsInstance(metric, abstract_metric.AbstractMetric)


class MetricResultTest(tf.test.TestCase, parameterized.TestCase):

  def test_create_and_validate_raises_error_if_shape_mismatch(self):
    with self.assertRaises(AssertionError):
      abstract_metric.MetricResult.create_and_validate(
          jnp.zeros((3, 1), dtype=jnp.float32),
          jnp.zeros((2, 1), dtype=jnp.bool_),
      )

  @parameterized.named_parameters(
      ('value_int', jnp.int32, jnp.bool_),
      ('valid_float', jnp.float32, jnp.float32),
  )
  def test_create_and_validate_raises_error_if_incorrect_dtype(
      self, value_dtype, valid_dtype
  ):
    shape = (1, 2)
    with self.assertRaises(AssertionError):
      abstract_metric.MetricResult.create_and_validate(
          jnp.zeros(shape, dtype=value_dtype),
          jnp.zeros(shape, dtype=valid_dtype),
      )

  def test_shape(self):
    shape = (3, 2)
    result = abstract_metric.MetricResult.create_and_validate(
        jnp.zeros(shape, dtype=jnp.float32), jnp.zeros(shape, dtype=jnp.bool_)
    )
    self.assertAllEqual(result.shape, shape)


if __name__ == '__main__':
  tf.test.main()
