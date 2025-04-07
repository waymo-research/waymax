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

"""Datastructure and helper functions for Waymax Observation functions."""
from typing import Any, Optional

import chex
import jax
import jax.numpy as jnp
import numpy as onp

from waymax import config
from waymax.datatypes import array
from waymax.datatypes import object_state
from waymax.datatypes import operations
from waymax.datatypes import roadgraph
from waymax.datatypes import route
from waymax.datatypes import simulator_state
from waymax.datatypes import traffic_lights
from waymax.utils import geometry


@chex.dataclass
class ObjectPose2D:
  """Object 2D pose that can be used for coordinate transformation.

  The pose information is stored with two format
  (see details at https://en.wikipedia.org/wiki/Pose_(computer_vision)):
  1) the position (original_xy) and orientation (original_yaw); 2) the
  transformation matrix (matrix) with the rotation angle explicitly saved as
  delta_yaw.

  Note since this is a general data structure, the pose could be using arbitrary
  coordinate system.

  Example usage:
  Assuming the pose is using C coordinate system, applying matrix[..., i, :, :]
  over an object with center (x, y) and yaw in C coordinate system gives its
  position in i's local coordinate system. The object's yaw in i's local
  coordinate system is yaw + delta_yaw.
  This is done by translation of -original_xy followed by counter-clockwise
  rotation of original_yaw.

  Attributes:
    original_xy: (..., num_objects, 2), the coordinates of each object's center
      in original coordinate systems.
    original_yaw: (..., num_objects), object's yaw in original coordinate
      systems.
    matrix: (..., num_objects, 3, 3), transformation matrix where `matrix[...,
      i, :, :]` is the transformation from original coordinates to object `i`
      centric coordinates.
    delta_yaw: (..., num_objects), rotation angles used to transform yaw in
      original coordinate system to object-center coordinate system. Note this
      is the negative of original_yaw.
    valid: (..., num_objects), valid or not.
  """

  original_xy: jax.Array
  original_yaw: jax.Array
  matrix: jax.Array
  delta_yaw: jax.Array
  valid: jax.Array

  @property
  def shape(self) -> tuple[int, ...]:
    """The Array shape."""
    return self.delta_yaw.shape

  @property
  def num_objects(self) -> int:
    """The number of objects."""
    return self.shape[-1]

  @property
  def rotation_matrix(self) -> jax.Array:
    """2D rotation matrix."""
    return self.matrix[..., :2, :2]

  @property
  def translation_vector(self) -> jax.Array:
    """2D translation vector."""
    return self.matrix[..., :2, 2]

  def __eq__(self, other: Any) -> bool:
    return operations.compare_all_leaf_nodes(self, other)

  @classmethod
  def from_center_and_yaw(
      cls, xy: jax.Array, yaw: jax.Array, valid: Optional[jax.Array] = None
  ) -> 'ObjectPose2D':
    """Initializes pose from center location and yaw.

    Args:
      xy: (..., num_objects, 2), 2D coordinates of objects' center in arbitrary
        coordinate system.
      yaw: (..., num_objects), objects' yaw in same coordinate system as xy.
      valid: (..., num_objects), boolean mask for validity.

    Returns:
      A ObjectPose2D with shape (..., num_objects).
    """
    pose_matrix, pose_yaw = geometry.pose_from_xy_yaw(xy, yaw)
    if valid is None:
      valid = jnp.ones_like(yaw, dtype=bool)
    out = cls(
        original_xy=xy,
        original_yaw=yaw,
        matrix=pose_matrix,
        delta_yaw=pose_yaw,
        valid=valid,
    )
    out.validate()
    return out

  @classmethod
  def from_transformation(
      cls,
      matrix: jax.Array,
      delta_yaw: jax.Array,
      valid: Optional[jax.Array] = None,
  ) -> 'ObjectPose2D':
    """Init pose from transformation matrix and delta_yaw.

    Args:
      matrix: (..., num_objects, 3, 3), 2D homogenous transformation matrix.
      delta_yaw: (..., num_objects), transformation angles used together with
        matrix for coordinate transformation (i.e. rotation applied by the
        transformation matrix).
      valid: (..., num_objects), boolean mask for validity.

    Returns:
      A ObjectPose2D with shape (..., num_objects).
    """
    prefix_shape = matrix.shape[:-2]
    xy = geometry.transform_points(
        jnp.linalg.inv(matrix), jnp.zeros((*prefix_shape, 2))
    )
    if valid is None:
      valid = jnp.ones_like(delta_yaw, dtype=bool)
    out = cls(
        original_xy=xy,
        original_yaw=-delta_yaw,
        matrix=matrix,
        delta_yaw=delta_yaw,
        valid=valid,
    )
    out.validate()
    return out

  def validate(self):
    """Validates shape and type."""
    chex.assert_equal_shape_prefix(
        [
            self.original_xy,
            self.original_yaw,
            self.matrix,
            self.delta_yaw,
            self.valid,
        ],
        len(self.shape),
    )
    chex.assert_type(
        [
            self.original_xy,
            self.original_yaw,
            self.matrix,
            self.delta_yaw,
            self.valid,
        ],
        [jnp.float32, jnp.float32, jnp.float32, jnp.float32, jnp.bool_],
    )


@chex.dataclass
class Observation:
  """Observation at a single simulation step.

  num_observations: number of objects that will have an observation/view over
  other objects/map elements.

  The observation can include a fixed number of history information. Note we
  support multi-agent observation by default: for each object (in axis
  num_observations), we compute its view of all other objects (in axis
  num_objects), roadgraph points, and traffic lights. The coordinates used for
  each object (num_observations) are centered at its location defined by pose2d.

  Attributes:
    trajectory: Time-dependent information, in object-centric coordinates
      defined by pose2d of shape (..., num_observations, num_objects,
      num_timesteps).
    is_ego: Binary mask of shape (..., num_observations, num_objects). It
      represents which object in num_objects is the observer itself.
    pose2d: Poses for all objects, used for transformation of shape (...,
      num_observations).
    metadata: Time-independent information of shape (..., num_observations,
      num_objects).
    roadgraph_static_points: Top-k (k=num_points) nearest static roadgraph
      points of shape (..., num_observations, num_points).
    traffic_lights: Current state of the traffic lights in the log of shape
      (..., num_observations, num_traffic_lights, num_timesteps).
    sdc_paths: SDC roadgraph traversal paths. Only valid for SDC agents of shape
      (..., num_observations, num_paths, num_points_per_path).
  """

  # Note num_observations represents the number of objects that we are
  # interested in (thus computing observation), while num_objects represents
  # the total number of objects in the scene. As an example:
  # trajectory[..., i, j, t] represents object i's view of object j at time t.
  trajectory: object_state.Trajectory
  is_ego: jax.Array
  pose2d: ObjectPose2D
  metadata: object_state.ObjectMetadata
  roadgraph_static_points: roadgraph.RoadgraphPoints
  traffic_lights: traffic_lights.TrafficLights
  sdc_paths: Optional[route.Paths]

  @property
  def shape(self) -> tuple[int, ...]:
    """The longest common prefix shape of all attributes: (..., num_observations)."""
    return self.pose2d.shape

  @property
  def batch_dims(self) -> tuple[int, ...]:
    """Batch dimensions."""
    return self.shape[:-1]

  @property
  def num_objects(self) -> int:
    """The number of objects."""
    return self.shape[-1]

  @property
  def valid(self) -> jax.Array:
    """Whether the observation is valid, (..., num_observations)."""
    return self.pose2d.valid

  def for_obj(self, idx: int) -> 'Observation':
    """Observation from idx-th object's point of view over all objects.

    Args:
      idx: The index of the object.

    Returns:
      A Observation with shape (..., 1).
    """
    axis = len(self.shape) - 2
    return operations.dynamic_slice(
        inputs=self, start_index=idx, slice_size=1, axis=axis
    )

  def validate(self):
    """Validates shape and type."""
    chex.assert_equal_shape_prefix(
        [
            self.trajectory,
            self.pose2d,
            self.metadata,
            self.roadgraph_static_points,
            self.traffic_lights,
        ],
        len(self.shape),
    )
    chex.assert_equal_shape([self.metadata, self.is_ego])
    chex.assert_type(self.is_ego, jnp.bool_)
    rank = len(self.shape)
    chex.assert_rank(self.trajectory, rank + 2)
    chex.assert_rank(self.pose2d, rank)
    chex.assert_rank(self.metadata, rank + 1)
    chex.assert_rank(self.roadgraph_static_points, rank + 1)
    chex.assert_rank(self.traffic_lights, rank + 2)
    if self.sdc_paths is not None:
      chex.assert_equal_shape_prefix(
          [self.trajectory, self.sdc_paths], len(self.shape)
      )
      chex.assert_rank(self.sdc_paths, rank + 2)


def transform_trajectory(
    traj: object_state.Trajectory, pose2d: ObjectPose2D
) -> object_state.Trajectory:
  """Transforms a Trajectory into coordinates specified by pose2d.

  Args:
    traj: A Trajectory with shape (..., num_observations, num_objects,
      num_timesteps) in coordinate system same as the one used by
      original_xy/yaw in pose2d.
    pose2d: A ObjectPose2D with shape (..., num_observations).

  Returns:
    A transformed trajectory in local coordinates per-observation defined by the
    pose.
  """
  chex.assert_equal_shape_prefix([traj, pose2d], len(pose2d.shape))

  local_xy = geometry.transform_points(
      pts=traj.xy,
      pose_matrix=pose2d.matrix,
  )
  local_vel_xy = geometry.transform_direction(
      pts_dir=traj.vel_xy,
      pose_matrix=pose2d.matrix,
  )
  # Expand pose2d dimensions to be compatible with input traj and computes the
  # transformed yaw and valid field.
  expand_axis = onp.arange(len(traj.shape) - len(pose2d.shape)) + len(
      pose2d.shape
  )
  local_yaw = traj.yaw + jnp.expand_dims(pose2d.delta_yaw, axis=expand_axis)
  local_valid = traj.valid & jnp.expand_dims(pose2d.valid, axis=expand_axis)

  return traj.replace(
      x=local_xy[..., 0],
      y=local_xy[..., 1],
      vel_x=local_vel_xy[..., 0],
      vel_y=local_vel_xy[..., 1],
      yaw=local_yaw,
      valid=local_valid,
  )


def transform_roadgraph_points(
    roadgraph_points: roadgraph.RoadgraphPoints, pose2d: ObjectPose2D
) -> roadgraph.RoadgraphPoints:
  """Transform a RoadgraphPoints into coordinates specified by pose2d.

  Args:
    roadgraph_points: A RoadgraphPoints with shape (..., num_observations,
      num_points).
    pose2d: A ObjectPose2D with shape (..., num_observations).

  Returns:
    A transformed RoadgraphPoints in local coordinates per-observation defined
    by the pose.
  """
  chex.assert_equal(
      roadgraph_points.shape, pose2d.shape + (roadgraph_points.shape[-1],)
  )

  dst_xy = geometry.transform_points(
      pts=roadgraph_points.xy,
      pose_matrix=pose2d.matrix,
  )
  dst_dir_xy = geometry.transform_direction(
      pts_dir=roadgraph_points.dir_xy, pose_matrix=pose2d.matrix
  )
  valid = jnp.logical_and(roadgraph_points.valid, pose2d.valid[..., None])

  return roadgraph_points.replace(
      x=dst_xy[..., 0],
      y=dst_xy[..., 1],
      dir_x=dst_dir_xy[..., 0],
      dir_y=dst_dir_xy[..., 1],
      valid=valid,
  )


def transform_traffic_lights(
    tls: traffic_lights.TrafficLights, pose2d: ObjectPose2D
) -> traffic_lights.TrafficLights:
  """Transforms a TrafficLightStates into coordinates specified by pose2d.

  Args:
    tls: A TrafficLightStates with shape (..., num_observations,
      num_traffic_lights, num_timesteps).
    pose2d: A ObjectPose2D with shape (..., num_observations).

  Returns:
    Transformed TrafficLightStates in local coordinates per-observation defined
    by the pose.
  """
  chex.assert_scalar_non_negative(len(tls.shape) - 2)
  chex.assert_equal(tls.shape, pose2d.shape + tls.shape[-2:])

  dst_xy = geometry.transform_points(
      pts=tls.xy,
      pose_matrix=pose2d.matrix,
  )
  valid = jnp.logical_and(tls.valid, pose2d.valid[..., None, None])

  return tls.replace(x=dst_xy[..., 0], y=dst_xy[..., 1], valid=valid)


def transform_observation(
    observation: Observation, pose2d: ObjectPose2D
) -> Observation:
  """Transforms a Observation into coordinates specified by pose2d.

  Args:
    observation: Has shape (..., num_observations)
    pose2d: Has shape (..., num_observations)

  Returns:
    Transformed observation in local coordinates per-observation defined by the
    pose.
  """
  # TODO(b/251216922) Potentially removes pose2d and lets user handle it.
  chex.assert_equal_shape([observation, pose2d])

  # Chain two rigid transformation using pose information.
  pose = combine_two_object_pose_2d(
      src_pose=observation.pose2d, dst_pose=pose2d
  )
  transf_traj = transform_trajectory(observation.trajectory, pose)
  transf_rg = transform_roadgraph_points(
      observation.roadgraph_static_points, pose
  )
  transf_tls = transform_traffic_lights(observation.traffic_lights, pose)

  obs = observation.replace(
      trajectory=transf_traj,
      roadgraph_static_points=transf_rg,
      traffic_lights=transf_tls,
      pose2d=pose2d,
  )
  obs.validate()
  return obs


def combine_two_object_pose_2d(
    src_pose: ObjectPose2D, dst_pose: ObjectPose2D
) -> ObjectPose2D:
  """Combines two ObjectPose2D as inverse(src_pose) plus dst_pose.

  Applying transformation using the returned pose is equivalent to applying
  transformation first with inverse(src_pose) and then dst_pose. Note as data
  transformation is much more expensive than computing the combined pose, it's
  more efficient to apply one data transformation with the combined pose instead
  of applying multiple transformations with multiple poses.

  Args:
    src_pose: The source pose.
    dst_pose: The destination/target pose.

  Returns:
    The combined pose.
  """
  return ObjectPose2D.from_transformation(
      matrix=jnp.matmul(
          dst_pose.matrix, jnp.linalg.inv(src_pose.matrix), precision='float32'
      ),
      delta_yaw=dst_pose.delta_yaw - src_pose.delta_yaw,
      valid=dst_pose.valid & src_pose.valid,
  )


@jax.named_scope('global_observation_from_state')
def global_observation_from_state(
    state: simulator_state.SimulatorState,
    obs_num_steps: int = 1,
    num_obj: int = 1,
) -> Observation:
  """Generates observation in global coordinates.

  Args:
    state: Has shape (...,).
    obs_num_steps: Number of observation steps for trajectories and traffic
      lights state.
    num_obj: Used to tile the global observation for multiple objects.

  Returns:
    Observation with shape (..., num_obj). Note the same observation in
    global coordinates is tiled for num_obj when num_obj is larger than 1.
  """

  def slice_by_time(x: operations.PyTree) -> operations.PyTree:
    return operations.dynamic_slice(
        x, state.timestep - obs_num_steps + 1, obs_num_steps, axis=-1
    )

  # Shape: (..., num_obj, num_timesteps) -> (..., num_obj, num_timesteps=obs_T).
  global_traj = slice_by_time(state.sim_trajectory)

  # Shape: (..., num_points).
  global_rg = state.roadgraph_points

  # [..., num_tls, T] --> [..., num_tls, obs_T]
  # Shape: (..., num_traffic_lights, num_timesteps) ->
  # (..., num_traffic_lights, num_timesteps=obs_T).
  global_tls = slice_by_time(state.log_traffic_light)
  metadata = state.object_metadata

  # Global observation has zero poses.
  pose2d_shape = state.shape
  pose2d = ObjectPose2D.from_center_and_yaw(
      xy=jnp.zeros(shape=pose2d_shape + (2,)),
      yaw=jnp.zeros(shape=pose2d_shape),
      valid=jnp.ones(shape=pose2d_shape, dtype=jnp.bool_),
  )

  # Agent-agnostic observation does not have SDC paths by default.
  sdc_paths_shape = state.shape + (1, 1)
  sdc_paths = route.Paths(
      x=jnp.zeros(shape=sdc_paths_shape),
      y=jnp.zeros(shape=sdc_paths_shape),
      z=jnp.zeros(shape=sdc_paths_shape),
      ids=jnp.zeros(shape=sdc_paths_shape, dtype=jnp.int32),
      valid=jnp.zeros(shape=sdc_paths_shape, dtype=jnp.bool_),
      arc_length=jnp.zeros(shape=sdc_paths_shape),
      on_route=jnp.zeros(shape=sdc_paths_shape, dtype=jnp.bool_),
  )

  global_obs = Observation(
      trajectory=global_traj,
      pose2d=pose2d,
      metadata=metadata,
      roadgraph_static_points=global_rg,
      traffic_lights=global_tls,
      sdc_paths=sdc_paths,
      is_ego=jnp.zeros(metadata.shape, dtype=jnp.bool_),  # Placeholder
  )

  obj_dim_idx = len(state.shape)
  global_obs_expanded = _tree_expand_and_repeat(
      global_obs, num_obj, obj_dim_idx
  )
  global_obs_expanded.validate()
  chex.assert_shape(global_obs_expanded, state.shape + (num_obj,))

  return global_obs_expanded


def _tree_expand_and_repeat(tree: Any, repeats: int, axis: int) -> array.PyTree:
  def _expand_and_repeat(x: jax.Array) -> jax.Array:
    return jnp.repeat(jnp.expand_dims(x, axis=axis), repeats, axis=axis)

  return jax.tree_util.tree_map(_expand_and_repeat, tree)


@jax.named_scope('observation_from_state')
def observation_from_state(
    state: simulator_state.SimulatorState,
    obs_num_steps: int = 1,
    roadgraph_top_k: int = 1000,
    coordinate_frame: config.CoordinateFrame = (config.CoordinateFrame.GLOBAL),
) -> Observation:
  """Constructs Observation from SimulatorState for all agents (jit-able).

  Args:
    state: A SimulatorState, with entirely variable shape (...).
    obs_num_steps: Number of steps history included in observation. Last
      timestep is state.timestep.
    roadgraph_top_k: Number of topk roadgraph observed by each object.
    coordinate_frame: Which coordinate frame the returned observation is using.

  Returns:
    Observation at current timestep from given simulator state, with shape (...,
    num_objects).
  """
  obj_xy = state.current_sim_trajectory.xy[..., 0, :]
  obj_yaw = state.current_sim_trajectory.yaw[..., 0]
  obj_valid = state.current_sim_trajectory.valid[..., 0]
  num_obj = state.sim_trajectory.num_objects

  global_obs = global_observation_from_state(state, obs_num_steps, num_obj)
  # Adds is_ego flags as one-hot vector, in this case it is computing
  # observation for all objects, thus it's a diagnal matrix with shape:
  # state.shape + (num_obj, num_obj)
  target_shape = state.shape + (num_obj, num_obj)
  is_ego = jax.lax.broadcast_in_dim(
      jnp.diag(jnp.arange(num_obj)) == 1,
      shape=target_shape,
      broadcast_dimensions=(len(target_shape) - 2, len(target_shape) - 1),
  )
  global_obs_filter = global_obs.replace(
      is_ego=is_ego,
      roadgraph_static_points=roadgraph.filter_topk_roadgraph_points(
          global_obs.roadgraph_static_points, obj_xy, roadgraph_top_k
      ),
  )

  if coordinate_frame == config.CoordinateFrame.OBJECT:
    pose2d = ObjectPose2D.from_center_and_yaw(
        xy=obj_xy, yaw=obj_yaw, valid=obj_valid
    )
  elif coordinate_frame == config.CoordinateFrame.GLOBAL:
    # No need to transform coordinates.
    return global_obs_filter
  elif coordinate_frame == config.CoordinateFrame.SDC:
    _, sdc_idx = jax.lax.top_k(state.object_metadata.is_sdc, k=1)
    sdc_xy = jnp.take_along_axis(obj_xy, sdc_idx[..., jnp.newaxis], axis=-2)
    sdc_yaw = jnp.take_along_axis(obj_yaw, sdc_idx, axis=-1)
    sdc_valid = jnp.take_along_axis(obj_valid, sdc_idx, axis=-1)
    # Replicates same pose for all objects since their observation are all using
    # SDC-centered coordinate system.
    sdc_xy_expand = jnp.repeat(sdc_xy, num_obj, axis=-2)
    sdc_yaw_expand = jnp.repeat(sdc_yaw, num_obj, axis=-1)
    sdc_valid_expand = jnp.repeat(sdc_valid, num_obj, axis=-1)
    pose2d = ObjectPose2D.from_center_and_yaw(
        xy=sdc_xy_expand, yaw=sdc_yaw_expand, valid=sdc_valid_expand
    )
  else:
    raise ValueError(f'Coordinate frame {coordinate_frame} not supported.')

  return transform_observation(global_obs_filter, pose2d)


@jax.named_scope('sdc_observation_from_state')
def sdc_observation_from_state(
    state: simulator_state.SimulatorState,
    obs_num_steps: int = 1,
    roadgraph_top_k: int = 1000,
    coordinate_frame: config.CoordinateFrame = (config.CoordinateFrame.SDC),
) -> Observation:
  """Constructs Observation from SimulatorState for SDC only (jit-able).

  Args:
    state: a SimulatorState, with shape (...)
    obs_num_steps: number of steps history included in observation. Last
      timestep is state.timestep.
    roadgraph_top_k: number of topk roadgraph observed by each object.
    coordinate_frame: which coordinate frame the returned observation is using.

  Returns:
    SDC Observation at current timestep from given simulator state, with shape
    (..., 1), where the last object dimension is 1 as there is only one SDC. It
    is not sequeezed to be consistent with multi-agent cases and compatible for
    other utils fnctions.
  """
  # Select the XY position at the current timestep.
  # Shape: (..., num_agents, 2)
  obj_xy = state.current_sim_trajectory.xy[..., 0, :]
  obj_yaw = state.current_sim_trajectory.yaw[..., 0]
  obj_valid = state.current_sim_trajectory.valid[..., 0]

  _, sdc_idx = jax.lax.top_k(state.object_metadata.is_sdc, k=1)
  sdc_xy = jnp.take_along_axis(obj_xy, sdc_idx[..., jnp.newaxis], axis=-2)
  sdc_yaw = jnp.take_along_axis(obj_yaw, sdc_idx, axis=-1)
  sdc_valid = jnp.take_along_axis(obj_valid, sdc_idx, axis=-1)

  # The num_obj is 1 because the it is computing the observation for SDC, and
  # there is only 1 SDC per scene.
  num_obj = 1
  global_obs = global_observation_from_state(
      state, obs_num_steps, num_obj=num_obj
  )
  is_ego = state.object_metadata.is_sdc[..., jnp.newaxis, :]
  global_obs_filter = global_obs.replace(
      is_ego=is_ego,
      roadgraph_static_points=roadgraph.filter_topk_roadgraph_points(
          global_obs.roadgraph_static_points, sdc_xy, roadgraph_top_k
      ),
  )

  if state.sdc_paths is not None:
    sdc_paths_expanded = _tree_expand_and_repeat(
        state.sdc_paths, num_obj, len(state.shape)
    )
    global_obs_filter = global_obs_filter.replace(sdc_paths=sdc_paths_expanded)
    global_obs_filter.validate()

  if coordinate_frame in (
      config.CoordinateFrame.OBJECT,
      config.CoordinateFrame.SDC,
  ):
    pose2d = ObjectPose2D.from_center_and_yaw(
        xy=sdc_xy, yaw=sdc_yaw, valid=sdc_valid
    )
    chex.assert_equal(pose2d.shape, state.shape + (1,))
    return transform_observation(global_obs_filter, pose2d)
  elif coordinate_frame == config.CoordinateFrame.GLOBAL:
    return global_obs_filter
  else:
    raise NotImplementedError(
        f'Coordinate frame {coordinate_frame} not supported.'
    )
