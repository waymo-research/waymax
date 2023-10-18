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

"""Tests for array data structures."""

from jax import numpy as jnp
import tensorflow as tf

from absl.testing import parameterized
from waymax.datatypes import array


class MaskedArrayTest(tf.test.TestCase, parameterized.TestCase):

  def test_create_and_validate_raises_error_if_incorrect_dtype(self):
    shape = (1,)
    with self.assertRaises(AssertionError):
      array.MaskedArray.create_and_validate(
          jnp.zeros(shape), jnp.zeros(shape, dtype=jnp.float32)
      )

  @parameterized.named_parameters(
      ('prefix_ndim1', (2,), (3,)),
      ('prefix_ndim2', (1, 2, 3), (1, 3)),
  )
  def test_create_and_validate_raises_error_if_incompatible_shape_prefix(
      self, value_shape, valid_shape
  ):
    with self.assertRaises(AssertionError):
      array.MaskedArray.create_and_validate(
          jnp.zeros(value_shape), jnp.zeros(valid_shape, dtype=jnp.bool_)
      )

  @parameterized.named_parameters(
      ('ndim1', (2,), (2,)),
      ('value_ndim2_valid_ndim1', (2, 3), (2,)),
      ('ndim2', (1, 2), (1, 2)),
      ('value_ndim3_valid_ndim2', (1, 2, 3), (1, 2)),
      ('value_ndim4_valid_ndim2', (1, 2, 3, 1), (1, 2)),
  )
  def test_create_and_validate_stores_value_and_valid_correctly(
      self, value_shape, valid_shape
  ):
    value = jnp.arange(jnp.prod(jnp.array(value_shape))).reshape(value_shape)
    valid = jnp.ones(valid_shape, dtype=jnp.bool_)
    masked_array = array.MaskedArray.create_and_validate(value, valid)
    with self.subTest('value'):
      self.assertAllEqual(masked_array.value, value)
    with self.subTest('valid'):
      self.assertAllEqual(masked_array.valid, valid)

  @parameterized.named_parameters(
      ('value_and_valid_same_shape', (2,), (2,)),
      ('value_and_valid_different_shapes', (2, 3), (2,)),
  )
  def test_shape(self, value_shape, valid_shape):
    value = jnp.arange(jnp.prod(jnp.array(value_shape))).reshape(value_shape)
    valid = jnp.ones(valid_shape, dtype=jnp.bool_)
    masked_array = array.MaskedArray.create_and_validate(value, valid)
    self.assertAllEqual(masked_array.shape, valid_shape)

  @parameterized.named_parameters(
      ('all_valid', [0, 1, jnp.inf], [True, True, True], 0, [0, 1, jnp.inf]),
      ('all_invalid', [0, 1, jnp.inf], [False, False, False], 0, [0, 0, 0]),
      ('partially_valid', [0, 1, jnp.inf], [True, True, False], 0, [0, 1, 0]),
      (
          'non_default_fill_value',
          [0, 1, jnp.inf],
          [False, True, False],
          3,
          [3, 1, 3],
      ),
      (
          'bool_fill_value',
          [0, 2, jnp.inf],
          [False, True, False],
          True,
          [1, 2, 1],
      ),
  )
  def test_mask(self, value, valid, fill_value, expected_result):
    masked_array = array.MaskedArray.create_and_validate(
        jnp.array(value, dtype=jnp.float32), jnp.array(valid, dtype=jnp.bool_)
    )
    result = masked_array.masked_value(fill_value)
    self.assertAllEqual(result, jnp.array(expected_result))


if __name__ == '__main__':
  tf.test.main()
