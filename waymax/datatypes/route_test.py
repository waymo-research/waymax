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

"""Tests for data_structures."""
from jax import numpy as jnp
import tensorflow as tf

from absl.testing import parameterized
from waymax.datatypes import route


class RouteDataStructTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.routes = route.Paths(
        x=jnp.array([0.0]),
        y=jnp.array([1.0]),
        z=jnp.array([2.0]),
        ids=jnp.array([3]),
        valid=jnp.array([True]),
        arc_length=jnp.array([4.0]),
        on_route=jnp.array([True]),
    )

  def test_route_paths_xy_returns_correctly(self):
    self.assertAllClose(self.routes.xy, jnp.array([[0.0, 1.0]]))

  def test_route_paths_xyz_returns_correctly(self):
    self.assertAllClose(self.routes.xyz, jnp.array([[0.0, 1.0, 2.0]]))

  def test_route_equality_works_correctly(self):
    with self.subTest('Equality=True'):
      self.assertEqual(self.routes, self.routes)
    with self.subTest('Equality=False'):
      self.assertNotEqual(self.routes, self.routes.replace(x=jnp.array([2.0])))

  def test_route_paths_validation_raises_when_necessary(self):
    with self.subTest('XWrongType'):
      error = 'input 0 has type int32 but expected .*float32.*.'
      with self.assertRaisesRegex(AssertionError, error):
        self.routes.replace(x=jnp.zeros((1), dtype=jnp.int32)).validate()
    with self.subTest('YWrongType'):
      error = 'input 1 has type int32 but expected .*float32.*.'
      with self.assertRaisesRegex(AssertionError, error):
        self.routes.replace(y=jnp.zeros((1), dtype=jnp.int32)).validate()
    with self.subTest('ZWrongType'):
      error = 'input 2 has type int32 but expected .*float32.*.'
      with self.assertRaisesRegex(AssertionError, error):
        self.routes.replace(z=jnp.zeros((1), dtype=jnp.int32)).validate()
    with self.subTest('IdsWrongType'):
      error = 'input 3 has type float32 but expected .*int32.*.'
      with self.assertRaisesRegex(AssertionError, error):
        self.routes.replace(ids=jnp.zeros((1), dtype=jnp.float32)).validate()
    with self.subTest('ValidWrongType'):
      error = 'input 4 has type int32 but expected .*bool.*.'
      with self.assertRaisesRegex(AssertionError, error):
        self.routes.replace(valid=jnp.zeros((1), dtype=jnp.int32)).validate()
    with self.subTest('ArcLengthWrongType'):
      error = 'input 5 has type int32 but expected .*float32.*.'
      with self.assertRaisesRegex(AssertionError, error):
        self.routes.replace(
            arc_length=jnp.zeros((1), dtype=jnp.int32)
        ).validate()
    with self.subTest('OnRouteWrongType'):
      error = 'input 6 has type int32 but expected .*bool.*.'
      with self.assertRaisesRegex(AssertionError, error):
        self.routes.replace(on_route=jnp.zeros((1), dtype=jnp.int32)).validate()


if __name__ == '__main__':
  tf.test.main()
