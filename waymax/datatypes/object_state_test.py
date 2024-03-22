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

import math

import jax
import jax.numpy as jnp
import tensorflow as tf

from absl.testing import parameterized
from waymax.dataloader import womd_factories
from waymax.datatypes import object_state
from waymax.utils import test_utils


class ObjectStateTest(tf.test.TestCase, parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.traj = object_state.Trajectory(
        x=jnp.array([0], dtype=jnp.float32),
        y=jnp.array([1], dtype=jnp.float32),
        z=jnp.array([2], dtype=jnp.float32),
        vel_x=jnp.array([3], dtype=jnp.float32),
        vel_y=jnp.array([4], dtype=jnp.float32),
        yaw=jnp.array([5], dtype=jnp.float32),
        valid=jnp.array([True], dtype=jnp.bool_),
        timestamp_micros=jnp.array([6], dtype=jnp.int32),
        length=jnp.array([7], dtype=jnp.float32),
        width=jnp.array([8], dtype=jnp.float32),
        height=jnp.array([9], dtype=jnp.float32),
    )
    cls.traj.validate()

  def test_fill_invalid(self):
    traj = object_state.Trajectory.zeros((5,))
    invalid_traj = object_state.fill_invalid_trajectory(traj)
    self.assertAllClose(invalid_traj.x, -jnp.ones((5,)))
    self.assertAllClose(invalid_traj.y, -jnp.ones((5,)))
    self.assertAllClose(invalid_traj.z, -jnp.ones((5,)))
    self.assertAllClose(invalid_traj.length, -jnp.ones((5,)))
    self.assertAllClose(invalid_traj.width, -jnp.ones((5,)))
    self.assertAllClose(invalid_traj.height, -jnp.ones((5,)))
    self.assertAllClose(invalid_traj.yaw, -jnp.ones((5,)))
    self.assertAllClose(invalid_traj.vel_x, -jnp.ones((5,)))
    self.assertAllClose(invalid_traj.vel_y, -jnp.ones((5,)))
    self.assertAllEqual(invalid_traj.valid, jnp.zeros((5,)))

  @parameterized.parameters(((),), ((2, 1),))
  def test_xyz_has_correct_shape(self, batch_dims):
    traj = object_state.Trajectory.zeros(shape=batch_dims)
    self.assertEqual(traj.xyz.shape, batch_dims + (3,))

  def test_vmap_func_works_for_traj(self):
    # `vmap`` creates empty objects that would fail if attributes shape and type
    # are checked in __pose_init__, thus we use validate function explicitly.
    dataset = test_utils.make_test_dataset(aggregate_timesteps=False)
    data_dict = next(dataset.as_numpy_iterator())
    traj = womd_factories.trajectory_from_womd_dict(data_dict, time_key='past')
    out = jax.vmap(lambda x: x)(traj)
    out.validate()

  def test_traj_bbox_corners(self):
    traj = object_state.Trajectory.zeros(shape=(1,))
    traj = traj.replace(
        width=jnp.ones(
            1,
        ),
        length=jnp.ones(
            1,
        ),
    )
    self.assertAllClose(
        traj.bbox_corners,
        jnp.array([[[0.5, -0.5], [0.5, 0.5], [-0.5, 0.5], [-0.5, -0.5]]]),
    )

  def test_object_metadata_shape_fields(self):
    value = jnp.zeros((10, 10, 100))
    object_metadata = object_state.ObjectMetadata(
        ids=value,
        object_types=value,
        is_sdc=value,
        is_modeled=value,
        is_valid=value,
        objects_of_interest=value,
        is_controlled=value,
    )
    self.assertEqual(object_metadata.shape, (10, 10, 100))
    self.assertEqual(object_metadata.num_objects, 100)

  def test_object_metadata_equality_works_properly(self):
    value = jnp.zeros((10, 10, 100))
    object_metadata = object_state.ObjectMetadata(
        ids=value,
        object_types=value,
        is_sdc=value,
        is_modeled=value,
        is_valid=value,
        objects_of_interest=value,
        is_controlled=value,
    )
    with self.subTest('Equalty=True'):
      self.assertEqual(object_metadata, object_metadata)
    with self.subTest('Equalty=False'):
      other_value = jnp.ones((10, 10, 100))
      other_metadata = object_state.ObjectMetadata(
          ids=other_value,
          object_types=other_value,
          is_sdc=other_value,
          is_modeled=other_value,
          is_valid=other_value,
          objects_of_interest=other_value,
          is_controlled=value,
      )
      self.assertNotEqual(object_metadata, other_metadata)

  def test_object_metadata_can_be_created_from_example_field(self):
    example = {
        'state/tracks_to_predict': jnp.array([0, 1, 0, -1]),
        'state/is_sdc': jnp.array([1, 0, 0, -1]),
        'state/id': jnp.array([0, 1, 2, -1]),
        'state/type': jnp.array([2, 0, 1, -1]),
        'state/objects_of_interest': jnp.array([0, 1, 1, -1]),
    }
    object_metadata = womd_factories.object_metadata_from_womd_dict(example)
    self.assertAllEqual(object_metadata.ids, jnp.array([0, 1, 2, -1]))
    self.assertAllEqual(object_metadata.object_types, jnp.array([2, 0, 1, -1]))
    self.assertAllEqual(
        object_metadata.is_sdc, jnp.array([True, False, False, False])
    )
    self.assertAllEqual(
        object_metadata.is_modeled, jnp.array([False, True, False, False])
    )
    self.assertAllEqual(
        object_metadata.is_valid, jnp.array([True, True, True, False])
    )
    self.assertAllEqual(
        object_metadata.objects_of_interest,
        jnp.array([False, True, True, False]),
    )

  def test_object_metadata_validate_asserts_if_improperly_created(self):
    object_metadata = object_state.ObjectMetadata(
        ids=jnp.zeros((1), dtype=jnp.int32),
        object_types=jnp.zeros((1), dtype=jnp.int32),
        is_sdc=jnp.zeros((1), dtype=jnp.bool_),
        is_modeled=jnp.zeros((1), dtype=jnp.bool_),
        is_valid=jnp.zeros((1), dtype=jnp.bool_),
        objects_of_interest=jnp.zeros((1), dtype=jnp.bool_),
        is_controlled=jnp.zeros((1), dtype=jnp.bool_),
    )

    with self.subTest('IdsWrongType'):
      object_metadata.validate()
      with self.assertRaises(AssertionError):
        object_metadata.replace(
            ids=jnp.zeros((1), dtype=jnp.float32)
        ).validate()
    with self.subTest('ObjectTypesWrongType'):
      object_metadata.validate()
      with self.assertRaises(AssertionError):
        object_metadata.replace(
            object_types=jnp.zeros((1), dtype=jnp.float32)
        ).validate()
    with self.subTest('IsSdcWrongType'):
      object_metadata.validate()
      with self.assertRaises(AssertionError):
        object_metadata.replace(
            is_sdc=jnp.zeros((1), dtype=jnp.float32)
        ).validate()
    with self.subTest('IsModeledWrongType'):
      object_metadata.validate()
      with self.assertRaises(AssertionError):
        object_metadata.replace(
            is_modeled=jnp.zeros((1), dtype=jnp.float32)
        ).validate()
    with self.subTest('ObjectMetadataWrongType'):
      object_metadata.validate()
      with self.assertRaises(AssertionError):
        object_metadata.replace(
            is_valid=jnp.zeros((1), dtype=jnp.float32)
        ).validate()
    with self.subTest('ObjectsOfInterestWrongType'):
      object_metadata.validate()
      with self.assertRaises(AssertionError):
        object_metadata.replace(
            objects_of_interest=jnp.zeros((1), dtype=jnp.float32)
        ).validate()

    with self.subTest('ShapesNotTheSame'):
      error = (
          '[Chex] Assertion assert_equal_shape failed: Arrays have '
          'different shapes: [(1,), (1,), (1,), (1,), (1, 1), (1,), (1,)].'
      )
      with self.assertRaisesWithLiteralMatch(AssertionError, error):
        object_metadata.replace(
            is_valid=jnp.zeros((1, 1), dtype=jnp.bool_)
        ).validate()

  def test_trajectory_shapes_are_correct(self):
    traj = object_state.Trajectory.zeros(shape=(1, 20, 91))
    self.assertEqual(traj.shape, (1, 20, 91))
    self.assertEqual(traj.num_objects, 20)
    self.assertEqual(traj.num_timesteps, 91)

  def test_traj_xy_returns_correct_values(self):
    self.assertAllClose(self.traj.xy, jnp.array([[0, 1]]))

  def test_traj_xyz_returns_correct_values(self):
    self.assertAllClose(self.traj.xyz, jnp.array([[0, 1, 2]]))

  def test_traj_vel_xy_returns_correct_values(self):
    self.assertAllClose(self.traj.vel_xy, jnp.array([[3, 4]]))

  def test_traj_speed_returns_correct_values(self):
    self.assertAllClose(self.traj.speed, jnp.array([5.0]))

  def test_traj_speed_returns_invalid_values_when_present(self):
    traj = self.traj.replace(
        vel_x=jnp.array([3.0, -1.0]),
        vel_y=jnp.array([4.0, -1.0]),
        valid=jnp.array([True, False]),
    )
    self.assertAllClose(traj.speed, jnp.array([5.0, -1.0]))

  def test_traj_vel_yaw_returns_correct_values(self):
    self.assertAllClose(self.traj.vel_yaw, jnp.array([0.9272952]))

  def test_traj_vel_yaw_returns_invalid_values_when_present(self):
    traj = self.traj.replace(
        vel_x=jnp.array([3.0, -1.0]),
        vel_y=jnp.array([4.0, -1.0]),
        valid=jnp.array([True, False]),
    )
    self.assertAllClose(traj.vel_yaw, jnp.array([0.9272952, -1.0]))

  def test_trajectory_equality_works_properly(self):
    with self.subTest('Equalty=True'):
      self.assertEqual(self.traj, self.traj)
    with self.subTest('Equalty=False'):
      self.assertNotEqual(self.traj, self.traj.replace(x=jnp.array([100.0])))

  def test_trajectory_stack_fields_returns_correct_values(self):
    traj_5dof = self.traj.stack_fields(['x', 'y', 'length', 'width', 'yaw'])
    self.assertAllClose(traj_5dof, jnp.array([[0, 1, 7, 8, 5]]))

  def test_bbox_corners_returns_proper_values(self):
    traj = self.traj.replace(
        x=jnp.array([0], dtype=jnp.float32),
        y=jnp.array([1], dtype=jnp.float32),
        yaw=jnp.array([math.pi / 2.0], dtype=jnp.float32),
        valid=jnp.array([True], dtype=jnp.bool_),
        length=jnp.array([1.0], dtype=jnp.float32),
        width=jnp.array([0.5], dtype=jnp.float32),
    )
    expected_bbox_corners = jnp.array(
        [[[0.25, 1.5], [-0.25, 1.5], [-0.25, 0.5], [0.25, 0.5]]]
    )
    self.assertAllClose(traj.bbox_corners, expected_bbox_corners)

  def test_zeros_returns_valid_datastructure(self):
    traj = object_state.Trajectory.zeros(shape=(10, 20, 100))
    # Make sure it's valid. Should not assert.
    traj.validate()
    zeros_traj = jax.tree_util.tree_map(jnp.zeros_like, traj)
    self.assertAllEqual(traj, zeros_traj)

  def test_trajectory_validate_asserts_if_improperly_created(self):
    with self.subTest('IdsWrongType'):
      with self.assertRaises(AssertionError):
        self.traj.replace(x=jnp.zeros((1), dtype=jnp.int32)).validate()
    with self.subTest('ObjectTypesWrongType'):
      with self.assertRaises(AssertionError):
        self.traj.replace(y=jnp.zeros((1), dtype=jnp.int32)).validate()
    with self.subTest('ObjectTypesWrongType'):
      with self.assertRaises(AssertionError):
        self.traj.replace(z=jnp.zeros((1), dtype=jnp.int32)).validate()
    with self.subTest('ObjectTypesWrongType'):
      with self.assertRaises(AssertionError):
        self.traj.replace(vel_x=jnp.zeros((1), dtype=jnp.int32)).validate()
    with self.subTest('ObjectTypesWrongType'):
      with self.assertRaises(AssertionError):
        self.traj.replace(vel_y=jnp.zeros((1), dtype=jnp.int32)).validate()
    with self.subTest('ObjectTypesWrongType'):
      with self.assertRaises(AssertionError):
        self.traj.replace(yaw=jnp.zeros((1), dtype=jnp.int32)).validate()
    with self.subTest('ObjectTypesWrongType'):
      with self.assertRaises(AssertionError):
        self.traj.replace(valid=jnp.zeros((1), dtype=jnp.float32)).validate()
    with self.subTest('ObjectTypesWrongType'):
      with self.assertRaises(AssertionError):
        self.traj.replace(
            timestamp_micros=jnp.zeros((1), dtype=jnp.float32)
        ).validate()
    with self.subTest('ObjectTypesWrongType'):
      with self.assertRaises(AssertionError):
        self.traj.replace(length=jnp.zeros((1), dtype=jnp.int32)).validate()
    with self.subTest('ObjectTypesWrongType'):
      with self.assertRaises(AssertionError):
        self.traj.replace(width=jnp.zeros((1), dtype=jnp.int32)).validate()
    with self.subTest('ObjectTypesWrongType'):
      with self.assertRaises(AssertionError):
        self.traj.replace(height=jnp.zeros((1), dtype=jnp.int32)).validate()

    with self.subTest('ShapesNotTheSame'):
      error = (
          '[Chex] Assertion assert_equal_shape failed: Arrays have '
          'different shapes: [(1,), (1, 1), (1,), (1,), (1,), (1,), '
          '(1,), (1,), (1,), (1,), (1,)].'
      )
      with self.assertRaisesWithLiteralMatch(AssertionError, error):
        self.traj.replace(y=jnp.zeros((1, 1), dtype=jnp.bool_)).validate()

  def test_width_length_height_are_the_same_over_time(self):
    shape_values = jnp.array([
        [1.0, 2.0, 1.0, 2.0, 1.0, 2.0, -1.0, -1, -1, -1],
        [2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, -1, -1],
        [4.0, 5.0, 4.0, 5.0, -1.0, -1.0, -1.0, -1, -1, -1],
    ])
    example = {
        'state/all/x': jnp.zeros((3, 10)),
        'state/all/y': jnp.zeros((3, 10)),
        'state/all/z': jnp.zeros((3, 10)),
        'state/all/velocity_x': jnp.zeros((3, 10)),
        'state/all/velocity_y': jnp.zeros((3, 10)),
        'state/all/bbox_yaw': jnp.zeros((3, 10)),
        'state/all/valid': shape_values != -1.0,
        'state/all/length': shape_values,
        'state/all/width': shape_values,
        'state/all/height': shape_values,
        'state/all/timestamp_micros': jnp.zeros((3, 10), dtype=jnp.int32),
    }
    traj = womd_factories.trajectory_from_womd_dict(example)
    expected_values = jnp.array([[1.5] * 10, [2.5] * 10, [4.5] * 10])
    self.assertAllClose(traj.width, expected_values)
    self.assertTrue(_check_values_are_all_same(traj.width, axis=-1))
    self.assertAllClose(traj.height, expected_values)
    self.assertTrue(_check_values_are_all_same(traj.height, axis=-1))
    self.assertAllClose(traj.length, expected_values)
    self.assertTrue(_check_values_are_all_same(traj.length, axis=-1))


def _check_values_are_all_same(value: jax.Array, axis: int = 0) -> jax.Array:
  """Checks to make sure all of the values reduced along axis are equal."""
  array_slice = [slice(dim) for dim in list(value.shape)]
  array_slice[axis] = 0
  tile_values = [1] * len(value.shape)
  tile_values[axis] = value.shape[axis]
  one_value_along_axis = jnp.tile(
      jnp.expand_dims(value[tuple(array_slice)], axis=axis), tile_values
  )
  return jnp.all(jnp.equal(one_value_along_axis, value))


if __name__ == '__main__':
  tf.test.main()
