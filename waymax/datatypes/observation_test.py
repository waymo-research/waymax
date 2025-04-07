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

import functools
import math
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
from waymax import config as _config
from waymax.dataloader import womd_dataloader
from waymax.dataloader import womd_factories
from waymax.datatypes import observation
from waymax.datatypes import operations
from waymax.datatypes import roadgraph
from waymax.datatypes import simulator_state
from waymax.utils import test_utils
from absl.testing import parameterized


class ObservationTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()

    self.config = _config.DatasetConfig(
        path=test_utils.ROUTE_DATA_PATH,
        max_num_objects=8,
        data_format=_config.DataFormat.TFRECORD,
    )
    state_t0 = next(womd_dataloader.simulator_state_generator(self.config))
    self.state_t10 = simulator_state.update_state_by_log(state_t0, 10)

    dataset = test_utils.make_test_dataset(batch_dims=(2, 1))

    tf_data_dict = dataset.take(1).get_single_element()
    data_dict = jax.tree_util.tree_map(jnp.asarray, tf_data_dict)
    self.default_traj = womd_factories.trajectory_from_womd_dict(data_dict)
    self.metadata = womd_factories.object_metadata_from_womd_dict(data_dict)

  @parameterized.parameters(((),), ((2, 1),))
  def test_pose2d_from_center_and_yaw_and_transformation_is_compatible(
      self, batch_dims
  ):
    num_obj = 32
    xy = jax.random.normal(
        jax.random.PRNGKey(0), shape=batch_dims + (num_obj, 2)
    )
    yaw = jax.random.normal(
        jax.random.PRNGKey(10), shape=batch_dims + (num_obj,)
    )
    pose2d_ = observation.ObjectPose2D.from_center_and_yaw(xy=xy, yaw=yaw)
    pose2d = observation.ObjectPose2D.from_transformation(
        matrix=pose2d_.matrix, delta_yaw=pose2d_.delta_yaw
    )

    self.assertAllClose(pose2d.original_xy, xy)
    self.assertAllClose(pose2d.original_yaw, yaw)

  def test_transform_traj_with_rotation_pi_and_2unit_translation(self):
    traj = jax.tree_util.tree_map(
        lambda x: jnp.ones(x.shape, x.dtype), self.default_traj
    )
    prefix_shape = traj.xy.shape[:-2]
    pose = observation.ObjectPose2D.from_center_and_yaw(
        xy=jnp.ones(prefix_shape + (2,), dtype=jnp.float32) * 2.0,
        yaw=jnp.ones(prefix_shape, dtype=jnp.float32) * jnp.pi,
    )
    # Note traj is a dummy test data with all value being 1.
    out = observation.transform_trajectory(traj, pose)

    self.assertAllClose(out.xy[0, 0, 0, 0], [1.0, 1.0])
    self.assertAllClose(out.vel_xy[0, 0, 0, 0], [-1.0, -1.0])
    self.assertAllClose(out.yaw[0, 0, 0, 0], -jnp.pi + 1.0)

  def test_transform_traj_runs_end2end_for_real_data(self):
    # Uses pose at timestep 0.
    pose = observation.ObjectPose2D.from_center_and_yaw(
        xy=self.default_traj.xy[..., 0, :], yaw=self.default_traj.yaw[..., 0]
    )
    out_traj = observation.transform_trajectory(self.default_traj, pose)

    flat_traj, _ = jax.tree_util.tree_flatten(self.default_traj)
    flat_out, _ = jax.tree_util.tree_flatten(out_traj)

    for traj, out in zip(flat_traj, flat_out):
      self.assertTupleEqual(traj.shape, out.shape)
      self.assertDTypeEqual(traj, out)

  @parameterized.parameters(((),), ((2, 1),))
  def test_transform_rg_pts_runs_end2end_for_real_data(self, batch_dims):
    dataset = test_utils.make_test_dataset(
        batch_dims=batch_dims, aggregate_timesteps=False
    )
    data_dict = next(dataset.as_numpy_iterator())
    traj = womd_factories.trajectory_from_womd_dict(data_dict, time_key='past')
    # All objects' pose at 5th timestep.
    pose2d = observation.ObjectPose2D.from_center_and_yaw(
        xy=traj.xy[..., 5, :], yaw=traj.yaw[..., 5]
    )
    rg_pts = womd_factories.roadgraph_from_womd_dict(data_dict)
    obj_dim_idx = rg_pts.x.ndim - 1

    def _expand_rg(x):
      return jnp.repeat(
          jnp.expand_dims(x, axis=obj_dim_idx),
          pose2d.num_objects,
          axis=obj_dim_idx,
      )

    rg_pts_with_obj_dim = jax.tree_util.tree_map(_expand_rg, rg_pts)
    out = observation.transform_roadgraph_points(rg_pts_with_obj_dim, pose2d)
    self.assertTupleEqual(
        out.shape, batch_dims + (pose2d.num_objects, rg_pts.num_points)
    )

  def test_combine_two_object_pose_2d(self):
    x, y, yaw = 1.0, 3**0.5, jnp.pi / 3
    dist = math.hypot(x, y)
    pose_a = observation.ObjectPose2D.from_center_and_yaw(
        xy=jnp.array([[x, y]]), yaw=jnp.array([yaw])
    )
    pose_b = observation.ObjectPose2D.from_center_and_yaw(
        xy=jnp.array([[dist, 0.0]]), yaw=jnp.array([yaw])
    )
    # Constructed such that applying transformation of pose_a followed by pose_b
    # is equivalent as applying pose_c.
    pose_c = observation.ObjectPose2D.from_center_and_yaw(
        xy=jnp.array([[x, y]]) * 2, yaw=jnp.array([yaw]) * 2
    )

    with self.subTest('pose_a_plus_b_equals_c'):
      pose_a_plus_b = observation.ObjectPose2D.from_transformation(
          matrix=jnp.matmul(pose_b.matrix, pose_a.matrix),
          delta_yaw=pose_a.delta_yaw + pose_b.delta_yaw,
      )
      self.assertTrue(
          operations.compare_all_leaf_nodes(pose_a_plus_b, pose_c, jnp.allclose)
      )
    with self.subTest('pose_c_minus_a_equals_b'):
      pose_c_minus_a = observation.combine_two_object_pose_2d(
          src_pose=pose_a, dst_pose=pose_c
      )
      self.assertTrue(
          operations.compare_all_leaf_nodes(
              pose_c_minus_a, pose_b, jnp.allclose
          )
      )

  def test_global_obs_from_state(self):
    global_obs = observation.global_observation_from_state(
        self.state_t10, obs_num_steps=2
    )
    log_traj = operations.dynamic_slice(
        self.state_t10.log_trajectory, start_index=9, slice_size=2, axis=-1
    )
    expected_traj = jax.tree_util.tree_map(
        lambda x: x[jnp.newaxis, ...], log_traj
    )
    expected_roadgraph = jax.tree_util.tree_map(
        lambda x: x[jnp.newaxis, ...], self.state_t10.roadgraph_points
    )
    log_tls = operations.dynamic_slice(
        self.state_t10.log_traffic_light, start_index=9, slice_size=2, axis=-1
    )
    expected_tls = jax.tree_util.tree_map(
        lambda x: x[jnp.newaxis, ...], log_tls
    )

    self.assertEqual(global_obs.shape, (1,))
    self.assertEqual(global_obs.trajectory, expected_traj)
    self.assertEqual(global_obs.roadgraph_static_points, expected_roadgraph)
    self.assertEqual(global_obs.traffic_lights, expected_tls)

  @parameterized.parameters(
      _config.CoordinateFrame.SDC,
      _config.CoordinateFrame.OBJECT,
      _config.CoordinateFrame.GLOBAL,
  )
  def test_obs_from_state_for_different_coordinate_system(
      self, coordinate_frame
  ):
    obs_from_state_fn = functools.partial(
        observation.observation_from_state,
        roadgraph_top_k=3,
        obs_num_steps=3,
        coordinate_frame=coordinate_frame,
    )
    obs = obs_from_state_fn(self.state_t10)

    with self.subTest('jit and non-jit gives same outputs'):
      jit_obs = jax.jit(obs_from_state_fn)(self.state_t10)
      self.assertAllEqual(jit_obs, obs)

    global_traj = operations.dynamic_slice(
        self.state_t10.log_trajectory, start_index=8, slice_size=3, axis=-1
    )
    global_tls = operations.dynamic_slice(
        self.state_t10.log_traffic_light, start_index=8, slice_size=3, axis=-1
    )
    sdc_idx = jnp.nonzero(self.state_t10.object_metadata.is_sdc)[0][0]
    sdc_pose = observation.ObjectPose2D.from_center_and_yaw(
        xy=self.state_t10.log_trajectory.xy[sdc_idx, 10],
        yaw=self.state_t10.log_trajectory.yaw[sdc_idx, 10],
        valid=self.state_t10.log_trajectory.valid[sdc_idx, 10],
    )

    self.assertEqual(obs.shape, (self.config.max_num_objects,))
    for i in range(self.config.max_num_objects):
      global_rg = roadgraph.filter_topk_roadgraph_points(
          self.state_t10.roadgraph_points,
          self.state_t10.log_trajectory.xy[i, 10],
          topk=3,
      )

      if coordinate_frame == _config.CoordinateFrame.GLOBAL:
        exp_traj = global_traj
        exp_rg = global_rg
        exp_tls = global_tls
      else:
        if coordinate_frame == _config.CoordinateFrame.OBJECT:
          pose = observation.ObjectPose2D.from_center_and_yaw(
              xy=self.state_t10.log_trajectory.xy[i, 10],
              yaw=self.state_t10.log_trajectory.yaw[i, 10],
              valid=self.state_t10.log_trajectory.valid[i, 10],
          )
        elif coordinate_frame == _config.CoordinateFrame.SDC:
          pose = sdc_pose
        else:
          raise ValueError(
              f'coordinate_frame: {coordinate_frame} is not supported.'
          )

        exp_traj = observation.transform_trajectory(global_traj, pose)
        exp_rg = observation.transform_roadgraph_points(global_rg, pose)
        exp_tls = observation.transform_traffic_lights(global_tls, pose)

      # pylint: disable=cell-var-from-loop
      jax.tree.map(
          np.testing.assert_allclose,
          jax.tree.map(lambda x: x[i], obs.trajectory),
          exp_traj,
      )
      jax.tree.map(
          np.testing.assert_allclose,
          jax.tree.map(lambda x: x[i], obs.roadgraph_static_points),
          exp_rg,
      )
      jax.tree.map(
          np.testing.assert_allclose,
          jax.tree.map(lambda x: x[i], obs.traffic_lights),
          exp_tls,
      )

  @parameterized.parameters(
      _config.CoordinateFrame.SDC,
      _config.CoordinateFrame.OBJECT,
      _config.CoordinateFrame.GLOBAL,
  )
  def test_sdc_obs_from_state_for_different_coordinate_system(
      self, coordinate_frame
  ):
    obs_from_state_fn = functools.partial(
        observation.sdc_observation_from_state,
        roadgraph_top_k=3,
        obs_num_steps=3,
        coordinate_frame=coordinate_frame,
    )
    sdc_obs = obs_from_state_fn(self.state_t10)

    with self.subTest('jit and non-jit gives same outputs'):
      jit_sdc_obs = jax.jit(obs_from_state_fn)(self.state_t10)
      self.assertAllEqual(jit_sdc_obs, sdc_obs)

    global_traj = operations.dynamic_slice(
        self.state_t10.log_trajectory, start_index=8, slice_size=3, axis=-1
    )
    global_tls = operations.dynamic_slice(
        self.state_t10.log_traffic_light, start_index=8, slice_size=3, axis=-1
    )
    sdc_idx = jnp.nonzero(self.state_t10.object_metadata.is_sdc)[0][0]
    sdc_pose = observation.ObjectPose2D.from_center_and_yaw(
        xy=self.state_t10.log_trajectory.xy[sdc_idx, 10],
        yaw=self.state_t10.log_trajectory.yaw[sdc_idx, 10],
        valid=self.state_t10.log_trajectory.valid[sdc_idx, 10],
    )
    global_rg = roadgraph.filter_topk_roadgraph_points(
        self.state_t10.roadgraph_points,
        self.state_t10.log_trajectory.xy[sdc_idx, 10],
        topk=3,
    )

    self.assertEqual(sdc_obs.shape, (1,))
    if coordinate_frame == _config.CoordinateFrame.GLOBAL:
      exp_traj = global_traj
      exp_rg = global_rg
      exp_tls = global_tls
    elif coordinate_frame in (
        _config.CoordinateFrame.OBJECT,
        _config.CoordinateFrame.SDC,
    ):
      exp_traj = observation.transform_trajectory(global_traj, sdc_pose)
      exp_rg = observation.transform_roadgraph_points(global_rg, sdc_pose)
      exp_tls = observation.transform_traffic_lights(global_tls, sdc_pose)
    else:
      raise ValueError(
          f'coordinate_frame: {coordinate_frame} is not supported.'
      )

    # pylint: disable=cell-var-from-loop
    self.assertEqual(
        jax.tree_util.tree_map(lambda x: x[0], sdc_obs.trajectory), exp_traj
    )
    self.assertEqual(
        jax.tree_util.tree_map(lambda x: x[0], sdc_obs.roadgraph_static_points),
        exp_rg,
    )
    self.assertEqual(
        jax.tree_util.tree_map(lambda x: x[0], sdc_obs.traffic_lights), exp_tls
    )


if __name__ == '__main__':
  tf.test.main()
