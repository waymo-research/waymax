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

import jax.numpy as jnp
import tensorflow as tf

from absl.testing import parameterized
from waymax.datatypes import traffic_lights
from waymax.utils import test_utils

TEST_DATA_PATH = test_utils.ROUTE_DATA_PATH


class TrafficLightsDataStructTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.tls = traffic_lights.TrafficLights(
        x=jnp.array([0.0], jnp.float32),
        y=jnp.array([1.0], jnp.float32),
        z=jnp.array([2.0], jnp.float32),
        state=jnp.array([traffic_lights.TrafficLightStates.ARROW_GO.value]),
        lane_ids=jnp.array([4], jnp.int32),
        valid=jnp.array([True]),
    )

  def test_traffic_lights_xy_returns_correct_values(self):
    self.assertAllClose(self.tls.xy, jnp.array([[0.0, 1.0]]))

  def test_traffic_lights_equality_works_properly(self):
    with self.subTest('Equality=True'):
      self.assertEqual(self.tls, self.tls)

    with self.subTest('Equality=False'):
      self.assertNotEqual(self.tls, self.tls.replace(x=jnp.array([1.0])))

  def test_traffic_lights_validity_works_properly(self):
    with self.subTest('XWrongType'):
      with self.assertRaises(AssertionError):
        self.tls.replace(x=jnp.zeros((1), dtype=jnp.int32)).validate()
    with self.subTest('YWrongType'):
      with self.assertRaises(AssertionError):
        self.tls.replace(y=jnp.zeros((1), dtype=jnp.int32)).validate()
    with self.subTest('ZWrongType'):
      with self.assertRaises(AssertionError):
        self.tls.replace(z=jnp.zeros((1), dtype=jnp.int32)).validate()
    with self.subTest('StateWrongType'):
      with self.assertRaises(AssertionError):
        self.tls.replace(state=jnp.zeros((1), dtype=jnp.float32)).validate()
    with self.subTest('LaneIdsWrongType'):
      with self.assertRaises(AssertionError):
        self.tls.replace(lane_ids=jnp.zeros((1), dtype=jnp.float32)).validate()
    with self.subTest('ValidWrongType'):
      with self.assertRaises(AssertionError):
        self.tls.replace(valid=jnp.zeros((1), dtype=jnp.int32)).validate()

    with self.subTest('ShapesNotTheSame'):
      error = (
          '[Chex] Assertion assert_equal_shape failed: Arrays have '
          'different shapes: [(1,), (1, 1), (1,), (1,), (1,), (1,)].'
      )
      with self.assertRaisesWithLiteralMatch(AssertionError, error):
        self.tls.replace(y=jnp.zeros((1, 1), dtype=jnp.bool_)).validate()


if __name__ == '__main__':
  tf.test.main()
