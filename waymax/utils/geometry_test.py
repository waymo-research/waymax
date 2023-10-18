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

import jax.numpy as jnp
import numpy as np

import tensorflow as tf

from absl.testing import parameterized
from waymax.utils import geometry


class GeometryTest(parameterized.TestCase):

  @parameterized.parameters(((1,),), ((),), ((3, 5, 7),))
  def test_rotation_matrix_2d_shape(self, shape: tuple[int, ...]):
    yaw = jnp.ones(shape, dtype=jnp.float32)
    rotation_mat = geometry.rotation_matrix_2d(yaw)
    self.assertEqual(rotation_mat.shape, shape + (2, 2))

  @parameterized.named_parameters(
      dict(
          testcase_name='pi',
          yaw=jnp.pi,
          expected_value=[[-1.0, 0.0], [0.0, -1.0]],
      ),
      dict(
          testcase_name='0.5*pi',
          yaw=0.5 * jnp.pi,
          expected_value=[[0.0, -1.0], [1.0, 0.0]],
      ),
      dict(
          testcase_name='1.5*pi',
          yaw=1.5 * jnp.pi,
          expected_value=[[0.0, 1.0], [-1.0, 0.0]],
      ),
      dict(testcase_name='0', yaw=0.0, expected_value=[[1.0, 0.0], [0.0, 1.0]]),
  )
  def test_rotation_matrix_2d_matches_expected_value(self, yaw, expected_value):
    rotation_mat = geometry.rotation_matrix_2d(yaw)
    np.testing.assert_array_almost_equal(rotation_mat, expected_value)

  def test_batched_rotate_matches_expected_value(self):
    yaws = jnp.array([jnp.pi, 0.0], dtype=jnp.float32)
    rotation_mat = geometry.rotation_matrix_2d(yaws)
    expected_result_0 = jnp.array([[-1.0, 0.0], [0.0, -1.0]])
    expected_result_1 = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    expected_result = jnp.stack([expected_result_0, expected_result_1])
    np.testing.assert_array_almost_equal(rotation_mat, expected_result)

  def test_corners_from_bboxes(self):
    # [5]
    bbox = jnp.array([0.0, 0.0, 1.0, 3**0.5, math.pi / 6])
    # [4, 2]
    expected = jnp.array([
        [0.5 * 3**0.5, -0.5],
        [0.0, 1.0],
        [-0.5 * 3**0.5, 0.5],
        [0.0, -1.0],
    ])
    with self.subTest('single_bbox'):
      corners = geometry.corners_from_bbox(bbox)
      np.testing.assert_array_almost_equal(corners, expected)
    with self.subTest('bboxes'):
      bboxes = jnp.tile(bbox[None, None, None, :], (2, 1, 3, 1))
      expected = jnp.tile(expected[None, None, None, :, :], (2, 1, 3, 1, 1))
      corners = geometry.corners_from_bboxes(bboxes)
      np.testing.assert_array_almost_equal(corners, expected)

  def test_has_overlap(self):
    bbox_a = jnp.array([[
        [0.0, 0.0, 2.0, 1.0, math.pi / 4],
        [10.0, 10.0, 2.0, 1.0, math.pi / 4],
        [0.0, 0.0, 2.0, 2.0, 0.0],
        [2.0, 2.0, 2 * 2**0.5, 2 * 2**0.5, math.pi / 4],
    ]])
    bbox_b = jnp.array([[
        [0.0, 1.5, 2.0, 1.0, math.pi / 4],
        [10.0, 11.5, 2.0, 1.0, -math.pi / 4],
        [2.0, 2.0, 2 * 2**0.5, 2 * 2**0.5, math.pi / 4],
        [0.0, 0.0, 1.0, 1.0, 0.0],
    ]])
    out = geometry.has_overlap(bbox_a, bbox_b)
    np.testing.assert_array_equal(out, [[False, True, False, False]])

  def test_transform_trajectories(self):
    x, y, yaw = 1.0, 2.0, math.pi / 3

    traj = jnp.array([[
        [x, y, 1.0, 2.0, yaw],
        [x + math.cos(yaw), y + math.sin(yaw), 11.0, 22.0, -yaw],
    ]])
    pose_matrix, pose_yaw = geometry.pose_from_xy_yaw(
        jnp.array([[x, y]]), jnp.array([yaw])
    )

    with self.subTest('forward_traj'):
      out_traj = geometry.transform_trajectories(traj, pose_matrix, pose_yaw)
      np.testing.assert_array_almost_equal(
          out_traj,
          [[[0.0, 0.0, 1.0, 2.0, 0.0], [1.0, 0.0, 11.0, 22.0, -yaw * 2]]],
      )

    with self.subTest('backward_traj'):
      origin = geometry.transform_trajectories(
          out_traj, jnp.linalg.inv(pose_matrix), -pose_yaw
      )
      np.testing.assert_array_almost_equal(origin, traj)

    with self.subTest('7dof_traj'):
      vel = jnp.array([[[1.0, 3.0**0.5], [3.0**0.5, -1.0]]])
      traj_7dof = jnp.concatenate([traj, vel], axis=-1)
      out_7dof = geometry.transform_trajectories(
          traj_7dof, pose_matrix, pose_yaw
      )
      np.testing.assert_array_almost_equal(
          out_7dof,
          [[
              [0.0, 0.0, 1.0, 2.0, 0.0, 2.0, 0.0],
              [1.0, 0.0, 11.0, 22.0, -yaw * 2, 0.0, -2.0],
          ]],
      )

  def test_wrap_yaws_small_yaw(self):
    yaws = jnp.array([1.0])
    wrapped_yaws = geometry.wrap_yaws(yaws)
    np.testing.assert_array_almost_equal(yaws, wrapped_yaws)

  def test_wrap_yaws_large_yaw(self):
    with self.subTest('LargeNegativeYaw'):
      large_neg_yaws = jnp.array([-jnp.pi - 0.1])
      np.testing.assert_array_almost_equal(
          jnp.array([jnp.pi - 0.1]), geometry.wrap_yaws(large_neg_yaws)
      )

    with self.subTest('LargePositiveYaw'):
      large_pos_yaws = jnp.array([jnp.pi + 0.1])
      np.testing.assert_array_almost_equal(
          jnp.array([-jnp.pi + 0.1]), geometry.wrap_yaws(large_pos_yaws)
      )

  @parameterized.parameters(((),), ((3, 4),))
  def test_pairwise_overlap_shapes(self, batch_dims):
    xy = jnp.array([[100.0, 200.0], [200.0, 100.0], [100.0, 200.0]])
    xy = jnp.tile(xy, batch_dims + (1, 1))
    yaw = jnp.array([jnp.pi / 2.0, -jnp.pi / 2.0, jnp.pi])
    yaw = jnp.tile(yaw, batch_dims + (1,))
    length = jnp.array([1.0, 0.5, 1.0])
    length = jnp.tile(length, batch_dims + (1,))
    width = jnp.array([0.5, 1.0, 1.0])
    width = jnp.tile(width, batch_dims + (1,))

    traj_5dof = jnp.concatenate(
        [
            xy,
            length[..., jnp.newaxis],
            width[..., jnp.newaxis],
            yaw[..., jnp.newaxis],
        ],
        axis=-1,
    )
    overlap_matrix = geometry.compute_pairwise_overlaps(traj_5dof)
    self.assertEqual(overlap_matrix.shape, batch_dims + (3, 3))

  def test_metric_with_no_overlap(self):
    # Create 6 boxes away from the SDC which should all result in no overlap.
    # The agent is stored at the origin (0, 0)
    num_agents = 7
    xy = jnp.array([
        [0.0, 0.0],
        [0.0, 2.0],
        [-3.0, 1.0],
        [3.0, 1.0],
        [3.0, -2.0],
        [-2.0, -1.5],
        [0.0, -3.0],
    ])
    yaw = jnp.arange(num_agents) * 0.5
    width = jnp.ones((num_agents,), dtype=jnp.float32)
    length = jnp.ones((num_agents,), dtype=jnp.float32) * 2
    traj_5dof = jnp.concatenate(
        [
            xy,
            length[..., jnp.newaxis],
            width[..., jnp.newaxis],
            yaw[..., jnp.newaxis],
        ],
        axis=-1,
    )
    overlap_matrix = geometry.compute_pairwise_overlaps(traj_5dof)
    sdc_overlap = overlap_matrix[0]
    self.assertEqual(list(sdc_overlap), [False] * (num_agents))

  def test_metric_with_overlaps(self):
    # Create 6 boxes around the SDC which should all result in overlaps.
    # The agent is stored at the origin (0, 0)
    num_agents = 7
    xy = jnp.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [-1.0, 1.0],
        [1.0, 1.0],
        [1.0, -1.0],
        [-1.0, -1.0],
        [0.0, -1.0],
    ])
    yaw = jnp.arange(num_agents) * 0.5
    width = jnp.ones((num_agents,), dtype=jnp.float32)
    length = jnp.ones((num_agents,), dtype=jnp.float32) * 2
    traj_5dof = jnp.concatenate(
        [
            xy,
            length[..., jnp.newaxis],
            width[..., jnp.newaxis],
            yaw[..., jnp.newaxis],
        ],
        axis=-1,
    )
    overlap_matrix = geometry.compute_pairwise_overlaps(traj_5dof)
    sdc_overlap = overlap_matrix[0]
    self.assertEqual(list(sdc_overlap), [False] + [True] * (num_agents - 1))

  def test_transform_direction(self):
    x, y, yaw = 1.0, 2.0, math.pi / 3
    # (3, 3)
    pose_matrix, _ = geometry.pose_from_xy_yaw(
        jnp.array([x, y]), jnp.array(yaw)
    )
    pts_dir = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    # Should correspond to -math.pi / 3 and math.pi / 6
    exp_dir = jnp.array([[0.5, -math.sqrt(3) / 2], [math.sqrt(3) / 2, 0.5]])
    with self.subTest('no_batch_dim'):
      out = geometry.transform_direction(pose_matrix, pts_dir)
      np.testing.assert_array_almost_equal(out, exp_dir)
    with self.subTest('has_batch_dim'):
      # (batch=2, 3, 3)
      pose_matrix_batch = jnp.tile(pose_matrix[jnp.newaxis, ...], (2, 1, 1))
      # (batch=2, num_pts=2, 2)
      pts_dir_batch = jnp.tile(pts_dir[jnp.newaxis, ...], (2, 1, 1))
      # (batch=2, num_pts=2, 2)
      out_batch = geometry.transform_direction(pose_matrix_batch, pts_dir_batch)
      exp_dir_batch = jnp.tile(exp_dir[jnp.newaxis, ...], (2, 1, 1))
      np.testing.assert_array_almost_equal(out_batch, exp_dir_batch)


if __name__ == '__main__':
  tf.test.main()
