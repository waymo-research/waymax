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

"""Factory functions for converting tensors to Waymax internal data structures."""

import jax
from jax import numpy as jnp

from waymax.datatypes import object_state
from waymax.datatypes import operations
from waymax.datatypes import roadgraph
from waymax.datatypes import route
from waymax.datatypes import simulator_state
from waymax.datatypes import traffic_lights


def object_metadata_from_womd_dict(
    example: dict[str, jax.Array]
) -> object_state.ObjectMetadata:
  """Constructs object metadata from an Open Motion TFExample dictionary.

  Args:
    example: Mapping from feature name to array for data read from WOMD
      `tf.Example`'s in the format of specified in the WOMD website
      (https://waymo.com/open/data/motion/tfexample).

  Returns:
    Metadata for the objects in the scene read from the WOMD data example of
      shape (..., num_objects).
  """
  is_valid = jnp.asarray(example['state/tracks_to_predict'] != -1)
  is_modeled = jnp.asarray(example['state/tracks_to_predict'] == 1)
  is_sdc = jnp.asarray(example['state/is_sdc'] == 1)
  objects_of_interest = jnp.asarray(example['state/objects_of_interest'] == 1)
  ids = jnp.asarray(example['state/id'], dtype=jnp.int32)
  object_types = jnp.asarray(example['state/type'], dtype=jnp.int32)
  is_controlled = jnp.zeros_like(is_sdc, dtype=jnp.bool_)

  out = object_state.ObjectMetadata(
      ids=ids,
      object_types=object_types,
      is_valid=is_valid,
      is_modeled=is_modeled,
      is_sdc=is_sdc,
      objects_of_interest=objects_of_interest,
      is_controlled=is_controlled,
  )
  out.validate()
  return out


def roadgraph_from_womd_dict(
    example: dict[str, jax.Array], prefix_key: str = 'roadgraph_samples'
) -> roadgraph.RoadgraphPoints:
  """Constructs a point map from an Open Motion TFExample dictionary.

  Args:
    example: Dictionary which contains data from the WOMD tf.Examples with
      optional extra batch dimensions. These are defined at
      https://waymo.com/open/data/motion/tfexample.
    prefix_key: Prefix key for the roadgraph type in the example.

  Returns:
    RoadgraphPoints from the `example` scenario with shape (..., num_points).
  """
  if not prefix_key.endswith('/'):
    prefix_key += '/'
  xyz = jnp.asarray(example[prefix_key + 'xyz'], dtype=jnp.float32)
  dir_xyz = jnp.asarray(example[prefix_key + 'dir'], dtype=jnp.float32)
  # Note all singleton axis is removed.
  types = jnp.asarray(example[prefix_key + 'type'][..., 0], dtype=jnp.int32)
  ids = jnp.asarray(example[prefix_key + 'id'][..., 0], dtype=jnp.int32)
  valids = jnp.asarray(example[prefix_key + 'valid'][..., 0], dtype=jnp.bool_)
  out = roadgraph.RoadgraphPoints(
      x=xyz[..., 0],
      y=xyz[..., 1],
      z=xyz[..., 2],
      dir_x=dir_xyz[..., 0],
      dir_y=dir_xyz[..., 1],
      dir_z=dir_xyz[..., 2],
      types=types,
      ids=ids,
      valid=valids,
  )
  out.validate()
  return out


def paths_from_womd_dict(
    example: dict[str, jax.Array], prefix_key: str = 'path_samples'
) -> route.Paths:
  """Constructs paths from an Open Motion TFExample dictionary.

  Args:
    example: Mapping from feature name to array for data read from WOMD, with
      extra key/values pairs for route related data.
    prefix_key: Keys for indexing route data.

  Returns:
    Paths with shape (..., num_paths, num_points_per_path).
  """
  if not prefix_key.endswith('/'):
    prefix_key += '/'
  xyz = jnp.asarray(example[f'{prefix_key}xyz'], dtype=jnp.float32)
  ids = jnp.asarray(example[f'{prefix_key}id'], dtype=jnp.int32)
  valids = jnp.asarray(example[f'{prefix_key}valid'], dtype=jnp.bool_)
  arc_length = jnp.asarray(
      example[f'{prefix_key}arc_length'], dtype=jnp.float32
  )
  on_route = jnp.asarray(example[f'{prefix_key}on_route'], dtype=jnp.bool_)
  out = route.Paths(
      x=xyz[..., 0],
      y=xyz[..., 1],
      z=xyz[..., 2],
      ids=ids,
      valid=valids,
      arc_length=arc_length,
      on_route=on_route,
  )
  out.validate()
  return out


def simulator_state_from_womd_dict(
    example: dict[str, jax.Array],
    include_sdc_paths: bool = False,
    time_key: str = 'all',
) -> simulator_state.SimulatorState:
  """Constructs a simulator state from an aggregated WOMD data dict.

  Args:
    example: Mapping from feature name to array for data read from WOMD
      `tf.Example`'s in the format of specified in the WOMD website
      (https://waymo.com/open/data/motion/tfexample). The data, however, can be
      augmented with different time keys. We do this in our data loader, by
      aggregating all timesteps into an `all` key prefix.
    include_sdc_paths: Whether to include sdc based route paths when
      constructing the simulator state. This is useful for representing the
      route of the agent. A similar parameter must be used in the data loader to
      get consistent behavior.
    time_key: Time step information to gather from `example` (`past`, `current`,
      `future`, `all`).

  Returns:
    A simulator state constructed at the initial timestep to spec given the
      parameters.

  Raises:
    ValueError: If a `time_key` parameter is passed in that is not one of
      (`past`, `current`, `future`, `all`).
  """
  if time_key not in ['past', 'current', 'future', 'all']:
    raise ValueError(f'time_key {time_key} is not supported.')
  roadgraph_points = roadgraph_from_womd_dict(
      example, prefix_key='roadgraph_samples'
  )

  if include_sdc_paths:
    # (..., num_paths, num_pts_per_path)
    sdc_paths = paths_from_womd_dict(example, prefix_key='path_samples')
  else:
    sdc_paths = None

  traffic_light = traffic_lights_from_womd_dict(example, time_key=time_key)
  object_metadata = object_metadata_from_womd_dict(example)

  log_trajectory = trajectory_from_womd_dict(example, time_key=time_key)
  # Init with zeros and false.
  sim_trajectory = jax.tree_util.tree_map(
      lambda x: jnp.zeros(x.shape, x.dtype), log_trajectory
  )

  # Updates timestep 0 from log_trajectory to sim_trajectory for all objects
  # and all fields.
  sim_trajectory = operations.update_by_slice_in_dim(
      sim_trajectory,
      log_trajectory,
      inputs_start_idx=0,
      slice_size=1,
      axis=-1,
  )

  batch_dims = sim_trajectory.shape[:-2]
  out = simulator_state.SimulatorState(
      timestep=jnp.zeros(batch_dims, dtype=jnp.int32),
      roadgraph_points=roadgraph_points,
      sim_trajectory=sim_trajectory,
      log_trajectory=log_trajectory,
      log_traffic_light=traffic_light,
      object_metadata=object_metadata,
      sdc_paths=sdc_paths,
  )
  out.validate()
  return out


def trajectory_from_womd_dict(
    example: dict[str, jax.Array], time_key: str = 'all'
) -> object_state.Trajectory:
  """Constructs a trajectory from an Open Motion TFExample dictionary.

  Args:
    example: Mapping from feature name to array for data read from WOMD
      `tf.Example`'s in the format of specified in the WOMD website
      (https://waymo.com/open/data/motion/tfexample).
    time_key: Key which represents what time dimension to access from the state
      fields in the example: (i.e. past, current, future, all).

  Returns:
    Trajectory of all objects from the `example` scenario with shape (...,
    num_objects, num_timesteps).
  """
  prefix = f'state/{time_key}'

  valid = jnp.asarray(example[f'{prefix}/valid'], jnp.bool_)
  length = jnp.asarray(example[f'{prefix}/length'], jnp.float32)
  length = operations.masked_mean(length, valid=valid, axis=-1)
  width = jnp.asarray(example[f'{prefix}/width'], jnp.float32)
  width = operations.masked_mean(width, valid=valid, axis=-1)
  height = jnp.asarray(example[f'{prefix}/height'], jnp.float32)
  height = operations.masked_mean(height, valid=valid, axis=-1)

  traj_obj = object_state.Trajectory(
      x=jnp.asarray(example[f'{prefix}/x'], jnp.float32),
      y=jnp.asarray(example[f'{prefix}/y'], jnp.float32),
      z=jnp.asarray(example[f'{prefix}/z'], jnp.float32),
      vel_x=jnp.asarray(example[f'{prefix}/velocity_x'], jnp.float32),
      vel_y=jnp.asarray(example[f'{prefix}/velocity_y'], jnp.float32),
      yaw=jnp.asarray(example[f'{prefix}/bbox_yaw'], jnp.float32),
      valid=valid,
      length=length,
      width=width,
      height=height,
      timestamp_micros=jnp.asarray(
          example[f'{prefix}/timestamp_micros'], jnp.int32
      ),
  )
  traj_obj.validate()
  return traj_obj


def traffic_lights_from_womd_dict(
    example: dict[str, jax.Array], time_key: str = 'all'
) -> traffic_lights.TrafficLights:
  """Constructs a traffic light state from WOMD TFExample dictionary.

  Args:
    example: Mapping from feature name to array for data read from WOMD
      `tf.Example`'s in the format of specified in the WOMD website
      (https://waymo.com/open/data/motion/tfexample). The data, however, can be
      augmented with different time keys. We do this in our data loader, by
      aggregating all timesteps into an `all` key.
    time_key: Time step information to gather from traffic light (`past`,
      `current`, `future`, `all`).

  Returns:
    Traffic lights from the `example` scenario with shape (...,
    num_traffic_lights, num_timesteps).

  Raises:
    ValueError: If `time_key` is not part of the accepted values.
  """
  if time_key not in ['past', 'current', 'future', 'all']:
    raise ValueError(f'time_key {time_key} is not supported.')

  prefix_time_key = f'traffic_light_state/{time_key}'

  x = jnp.asarray(example[f'{prefix_time_key}/x'], dtype=jnp.float32)
  y = jnp.asarray(example[f'{prefix_time_key}/y'], dtype=jnp.float32)
  z = jnp.asarray(example[f'{prefix_time_key}/z'], dtype=jnp.float32)

  state = jnp.asarray(example[f'{prefix_time_key}/state'], dtype=jnp.int32)
  lane_ids = jnp.asarray(example[f'{prefix_time_key}/id'], dtype=jnp.int32)
  valid = jnp.asarray(example[f'{prefix_time_key}/valid'], dtype=jnp.bool_)

  origin = traffic_lights.TrafficLights(
      x=x, y=y, z=z, state=state, lane_ids=lane_ids, valid=valid
  )

  # New Shape: (..., num_traffic_lights, num_timesteps).
  out = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, -1, -2), origin)
  out.validate()
  return out
