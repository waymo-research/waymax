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

import jax
from jax import numpy as jnp
import tensorflow as tf

from absl.testing import parameterized
from waymax.datatypes import operations


class DataOperationsTest(tf.test.TestCase, parameterized.TestCase):

  def test_can_update_slice_of_arraytree_with_random_values(self):
    inputs = [
        jnp.reshape(jnp.arange(8), (2, 4)),
        jnp.reshape(jnp.arange(8) + 100.0, (2, 4)),
    ]
    updates = [
        jnp.reshape(jnp.arange(4), (2, 2)),
        jnp.reshape(jnp.arange(4) + 100.0, (2, 2)),
    ]
    out = operations.update_by_slice_in_dim(
        inputs,
        updates,
        inputs_start_idx=1,
        updates_start_idx=0,
        slice_size=2,
        axis=-1,
    )
    exp = [
        jnp.array([[0, 0, 1, 3], [4, 2, 3, 7]]),
        jnp.array([[100, 100, 101, 103], [104, 102, 103, 107]]),
    ]
    self.assertAllClose(out, exp)

  def test_update_by_mask_works_properly(self):
    inputs = {'one': jnp.arange(8), 'two': jnp.arange(9, 17)}
    updates = {
        'one': jnp.ones_like(inputs['one']) * 100.0,
        'two': jnp.ones_like(inputs['two']) * 200.0,
    }
    mask = jnp.array([True, False, False, True, False, False, False, True])
    out = operations.update_by_mask(inputs, updates, mask)
    exp = {
        'one': jnp.array([100.0, 1.0, 2.0, 100.0, 4.0, 5.0, 6.0, 100.0]),
        'two': jnp.array([200.0, 10.0, 11.0, 200.0, 13.0, 14.0, 15.0, 200.0]),
    }
    self.assertAllClose(out, exp)

  def test_select_by_onehot_has_correct_shapes(self):
    data = jnp.zeros((1, 2, 3, 4))
    selection = jnp.zeros((1, 2))
    result = operations.select_by_onehot(data, selection, keepdims=True)
    self.assertEqual(result.shape, (1, 1, 3, 4))
    result = operations.select_by_onehot(data, selection, keepdims=False)
    self.assertEqual(result.shape, (1, 3, 4))

  def test_select_by_onehot_has_correct_values(self):
    data = jnp.arange(8)
    selection = jnp.zeros(8).at[4].set(1.0)
    result = operations.select_by_onehot(data, selection, keepdims=True)
    self.assertEqual(result, data[4:5])

  def test_dynamic_slice_returns_correct_values(self):
    inps = {'one': jnp.arange(8), 'two': jnp.arange(9, 17)}
    out = operations.dynamic_slice(inps, start_index=3, slice_size=3)
    expected = {'one': jnp.array([3, 4, 5]), 'two': jnp.array([12, 13, 14])}
    self.assertAllClose(out, expected)

  def test_dynamic_slice_with_batched_index(self):
    inps = {'one': jnp.arange(8), 'two': jnp.arange(9, 17)}
    inps = jax.tree_util.tree_map(lambda x: jnp.stack([x, x], axis=0), inps)
    out = operations.dynamic_slice(
        inps, start_index=jnp.array([3, 3]), slice_size=3
    )
    expected = {'one': jnp.array([3, 4, 5]), 'two': jnp.array([12, 13, 14])}
    expected = jax.tree_util.tree_map(
        lambda x: jnp.stack([x, x], axis=0), expected
    )
    self.assertAllClose(out, expected)

  def test_dynamic_index_returns_correct_value_no_keepdims(self):
    inps = {
        'one': jnp.stack((jnp.arange(8), jnp.arange(9, 17)), axis=0),
        'two': jnp.stack((jnp.arange(18, 26), jnp.arange(27, 35)), axis=0),
    }
    out = operations.dynamic_index(inps, index=3, axis=-1, keepdims=False)
    expected = {'one': jnp.array([3, 12]), 'two': jnp.array([21, 30])}
    self.assertAllClose(out, expected)

  def test_dynamic_index_returns_correct_value_keepdims(self):
    inps = {
        'one': jnp.stack((jnp.arange(8), jnp.arange(9, 17)), axis=0),
        'two': jnp.stack((jnp.arange(18, 26), jnp.arange(27, 35)), axis=0),
    }
    out = operations.dynamic_index(inps, index=3, axis=-1, keepdims=True)
    expected = {'one': jnp.array([[3], [12]]), 'two': jnp.array([[21], [30]])}
    self.assertAllClose(out, expected)

  def test_compare_all_leaf_nodes_returns_true(self):
    inps = {
        'one': jnp.stack((jnp.arange(8), jnp.arange(9, 17)), axis=0),
        'two': jnp.stack((jnp.arange(18, 26), jnp.arange(27, 35)), axis=0),
    }
    self.assertTrue(operations.compare_all_leaf_nodes(inps, inps))

  def test_compare_all_leaf_nodes_returns_false(self):
    inps = {
        'one': jnp.stack((jnp.arange(8), jnp.arange(9, 17)), axis=0),
        'two': jnp.stack((jnp.arange(18, 26), jnp.arange(27, 35)), axis=0),
    }
    inps2 = {'one': inps['two'], 'two': inps['one']}
    self.assertFalse(operations.compare_all_leaf_nodes(inps, inps2))

  @parameterized.parameters(
      (jnp.bool_, False), (jnp.float32, -1.0), (jnp.int32, -1)
  )
  def test_invalid_values(self, dtype, expected_value):
    result = operations.make_invalid_data(jnp.zeros((2, 3), dtype=dtype))
    self.assertAllClose(result, jnp.full((2, 3), expected_value, dtype=dtype))

  def test_masked_mean_works_properly(self):
    values = jnp.array([
        [1.0, 2.0, 1.0, 2.0, 1.0, 2.0, -1.0, -1, -1, -1],
        [2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, -1, -1],
        [4.0, 5.0, 4.0, 5.0, -1.0, -1.0, -1.0, -1, -1, -1],
    ])
    expected_values = jnp.array([[1.5] * 10, [2.5] * 10, [4.5] * 10])
    self.assertAllClose(
        operations.masked_mean(values, values != -1.0, axis=-1), expected_values
    )

  def test_masked_mean_works_properly_if_no_valid(self):
    values = jnp.array([
        [1.0, 2.0, 1.0, 2.0, 1.0, 2.0, -1.0, -1, -1, -1],
        [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1, -1],
        [4.0, 5.0, 4.0, 5.0, -1.0, -1.0, -1.0, -1, -1, -1],
    ])
    expected_values = jnp.array([[1.5] * 10, [-1.0] * 10, [4.5] * 10])
    self.assertAllClose(
        operations.masked_mean(values, values != -1.0, axis=-1), expected_values
    )


if __name__ == '__main__':
  tf.test.main()
