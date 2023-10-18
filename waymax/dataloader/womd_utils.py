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

"""General settings and utility functions specifically for the WOMD data.

These functions are mainly intended to be iternal into the Waymax data library.

WOMD (go/womd) represents the data we typically use for simulation in our
environment. See https://waymo.com/open/data/motion/tfexample for definitions on
many of the data fields mentioned in this file.
"""
from typing import Optional

import jax
from jax.experimental import checkify
import jax.numpy as jnp
import tensorflow as tf
import tree

from waymax.datatypes import object_state
from waymax.datatypes import operations
from waymax.datatypes import roadgraph
from waymax.datatypes import simulator_state
from waymax.datatypes import traffic_lights

# Default values for WOMD different datatypes.
DEFAULT_FLOAT = -1.0
DEFAULT_INT = -1
DEFAULT_BOOL = False
# Dictionary converting a tensorflow data type to jax.numpy data types.
_TF_TO_JNP_DTYPE = {
    tf.int32: jnp.int32,
    tf.int64: jnp.int64,
    tf.float32: jnp.float32,
    tf.float64: jnp.float64,
    tf.bool: jnp.bool_,
}

# The num_steps axis for fields of the form
# traffic_light_state/xxx/timestamp_micros according to the specs at
# https://waymo.com/open/data/motion/tfexample
TL_TIMESTAMP_STEP_AXIS = -1
# The num_steps axis for all non-timestamp_micros traffic light fields.
TL_STEP_AXIS = -2


def aggregate_time_tensors(
    decoded_tensors: dict[str, tf.Tensor]
) -> dict[str, tf.Tensor]:
  """Combines all past/current/future fields into an 'all' field.

  Note the original past/current/future keys are removed in the returned dict.

  Args:
    decoded_tensors: input dict of tensors keyed by string.

  Returns:
    A new dict of tensors keyed by updated string: `past/current/future` are
    merged into `all`.
  """
  # Enumerate all the time tensors under state/.
  state_time_features, traffic_light_time_features = set(), set()
  for t_key in decoded_tensors:
    if t_key.startswith('state/current/'):
      state_time_features.add(t_key[len('state/current/') :])
    elif t_key.startswith('traffic_light_state/current/'):
      traffic_light_time_features.add(
          t_key[len('traffic_light_state/current/') :]
      )
  # All these feature are replaced by a single /all/ feature, a concatenation of
  # past, current, future
  removed_keys = set()
  concatenated_tensors = {}
  for state_feature in state_time_features:
    concatenated_tensors[f'state/all/{state_feature}'] = tf.concat(
        [
            decoded_tensors[f'state/past/{state_feature}'],
            decoded_tensors[f'state/current/{state_feature}'],
            decoded_tensors[f'state/future/{state_feature}'],
        ],
        axis=-1,
    )  # Shape is (num_objects, num_timesteps).
    removed_keys.add(f'state/past/{state_feature}')
    removed_keys.add(f'state/current/{state_feature}')
    removed_keys.add(f'state/future/{state_feature}')
  for traffic_light_feature in traffic_light_time_features:
    step_axis = (
        TL_TIMESTAMP_STEP_AXIS
        if traffic_light_feature == 'timestamp_micros'
        else TL_STEP_AXIS
    )
    concatenated_tensors[f'traffic_light_state/all/{traffic_light_feature}'] = (
        tf.concat(
            [
                decoded_tensors[
                    f'traffic_light_state/past/{traffic_light_feature}'
                ],
                decoded_tensors[
                    f'traffic_light_state/current/{traffic_light_feature}'
                ],
                decoded_tensors[
                    f'traffic_light_state/future/{traffic_light_feature}'
                ],
            ],
            axis=step_axis,
        )
    )  # Shape is (num_timesteps, num_traffic lights).
    removed_keys.add(f'traffic_light_state/past/{traffic_light_feature}')
    removed_keys.add(f'traffic_light_state/current/{traffic_light_feature}')
    removed_keys.add(f'traffic_light_state/future/{traffic_light_feature}')
  # We also add an additional feature called which_time, -1 for past, 0 for
  # current and 1 for future. This is used internally to decide handovers.
  past_length = decoded_tensors['state/past/valid'].shape[1]
  future_length = decoded_tensors['state/future/valid'].shape[1]
  concatenated_tensors['state/which_time'] = tf.concat(
      [-tf.ones((past_length,)), tf.zeros((1,)), tf.ones((future_length,))],
      axis=0,
  )
  # Add all other features without a time (past/current/future) dimension.
  for t_key in decoded_tensors:
    if t_key not in removed_keys:
      concatenated_tensors[t_key] = decoded_tensors[t_key]
  return concatenated_tensors


def get_features_description(
    max_num_objects: int = 128,
    max_num_rg_points: int = 30000,
    include_sdc_paths: bool = False,
    num_paths: Optional[int] = 45,
    num_points_per_path: Optional[int] = 800,
    num_tls: Optional[int] = 16,
) -> dict[str, tf.io.FixedLenFeature]:
  """Returns a dictionary of all features to be extracted.

  Args:
    max_num_objects: Max number of objects.
    max_num_rg_points: Max number of sampled roadgraph points.
    include_sdc_paths: Whether to include roadgraph traversal paths for the SDC.
    num_paths: Optional number of SDC paths. Must be defined if
      `include_sdc_paths` is True.
    num_points_per_path: Optional number of points per SDC path. Must be defined
      if `include_sdc_paths` is True.
    num_tls: Maximum number of traffic lights.

  Returns:
    Dictionary of all features to be extracted.

  Raises:
    ValueError: If `include_sdc_paths` is True but either `num_paths` or
      `num_points_per_path` is None.
  """
  if include_sdc_paths and (num_paths is None or num_points_per_path is None):
    raise ValueError(
        'num_paths and num_points_per_path must be defined if SDC '
        'paths are included (include_sdc_paths).'
    )

  roadgraph_features = {
      'roadgraph_samples/dir': tf.io.FixedLenFeature(
          [max_num_rg_points, 3], tf.float32, default_value=None
      ),
      'roadgraph_samples/id': tf.io.FixedLenFeature(
          [max_num_rg_points, 1], tf.int64, default_value=None
      ),
      'roadgraph_samples/type': tf.io.FixedLenFeature(
          [max_num_rg_points, 1], tf.int64, default_value=None
      ),
      'roadgraph_samples/valid': tf.io.FixedLenFeature(
          [max_num_rg_points, 1], tf.int64, default_value=None
      ),
      'roadgraph_samples/xyz': tf.io.FixedLenFeature(
          [max_num_rg_points, 3], tf.float32, default_value=None
      ),
  }

  # Features of other agents.
  state_features = {
      'state/id': tf.io.FixedLenFeature(
          [max_num_objects], tf.float32, default_value=None
      ),
      'state/type': tf.io.FixedLenFeature(
          [max_num_objects], tf.float32, default_value=None
      ),
      'state/is_sdc': tf.io.FixedLenFeature(
          [max_num_objects], tf.int64, default_value=None
      ),
      'state/tracks_to_predict': tf.io.FixedLenFeature(
          [max_num_objects], tf.int64, default_value=None
      ),
      'state/objects_of_interest': tf.io.FixedLenFeature(
          [max_num_objects], tf.int64, default_value=None
      ),
  }
  num_timesteps = {'past': 10, 'current': 1, 'future': 80}
  for time in ['past', 'current', 'future']:
    steps_to_extract = num_timesteps[time]
    state_time_features = {
        'state/%s/bbox_yaw'
        % time: tf.io.FixedLenFeature(
            [max_num_objects, steps_to_extract], tf.float32, default_value=None
        ),
        'state/%s/height'
        % time: tf.io.FixedLenFeature(
            [max_num_objects, steps_to_extract], tf.float32, default_value=None
        ),
        'state/%s/length'
        % time: tf.io.FixedLenFeature(
            [max_num_objects, steps_to_extract], tf.float32, default_value=None
        ),
        'state/%s/timestamp_micros'
        % time: tf.io.FixedLenFeature(
            [max_num_objects, steps_to_extract], tf.int64, default_value=None
        ),
        'state/%s/valid'
        % time: tf.io.FixedLenFeature(
            [max_num_objects, steps_to_extract], tf.int64, default_value=None
        ),
        'state/%s/vel_yaw'
        % time: tf.io.FixedLenFeature(
            [max_num_objects, steps_to_extract], tf.float32, default_value=None
        ),
        'state/%s/speed'
        % time: tf.io.FixedLenFeature(
            [max_num_objects, steps_to_extract], tf.float32, default_value=None
        ),
        'state/%s/velocity_x'
        % time: tf.io.FixedLenFeature(
            [max_num_objects, steps_to_extract], tf.float32, default_value=None
        ),
        'state/%s/velocity_y'
        % time: tf.io.FixedLenFeature(
            [max_num_objects, steps_to_extract], tf.float32, default_value=None
        ),
        'state/%s/width'
        % time: tf.io.FixedLenFeature(
            [max_num_objects, steps_to_extract], tf.float32, default_value=None
        ),
        'state/%s/x'
        % time: tf.io.FixedLenFeature(
            [max_num_objects, steps_to_extract], tf.float32, default_value=None
        ),
        'state/%s/y'
        % time: tf.io.FixedLenFeature(
            [max_num_objects, steps_to_extract], tf.float32, default_value=None
        ),
        'state/%s/z'
        % time: tf.io.FixedLenFeature(
            [max_num_objects, steps_to_extract], tf.float32, default_value=None
        ),
    }
    state_features.update(state_time_features)

  traffic_light_features = {
      'traffic_light_state/current/state': tf.io.FixedLenFeature(
          [1, num_tls], tf.int64, default_value=None
      ),
      'traffic_light_state/current/valid': tf.io.FixedLenFeature(
          [1, num_tls], tf.int64, default_value=None
      ),
      'traffic_light_state/current/id': tf.io.FixedLenFeature(
          [1, num_tls], tf.int64, default_value=None
      ),
      'traffic_light_state/current/x': tf.io.FixedLenFeature(
          [1, num_tls], tf.float32, default_value=None
      ),
      'traffic_light_state/current/y': tf.io.FixedLenFeature(
          [1, num_tls], tf.float32, default_value=None
      ),
      'traffic_light_state/current/z': tf.io.FixedLenFeature(
          [1, num_tls], tf.float32, default_value=None
      ),
      'traffic_light_state/current/timestamp_micros': tf.io.FixedLenFeature(
          [
              1,
          ],
          tf.int64,
          default_value=None,
      ),
      'traffic_light_state/past/state': tf.io.FixedLenFeature(
          [10, num_tls], tf.int64, default_value=None
      ),
      'traffic_light_state/past/valid': tf.io.FixedLenFeature(
          [10, num_tls], tf.int64, default_value=None
      ),
      'traffic_light_state/past/id': tf.io.FixedLenFeature(
          [10, num_tls], tf.int64, default_value=None
      ),
      'traffic_light_state/past/x': tf.io.FixedLenFeature(
          [10, num_tls], tf.float32, default_value=None
      ),
      'traffic_light_state/past/y': tf.io.FixedLenFeature(
          [10, num_tls], tf.float32, default_value=None
      ),
      'traffic_light_state/past/z': tf.io.FixedLenFeature(
          [10, num_tls], tf.float32, default_value=None
      ),
      'traffic_light_state/past/timestamp_micros': tf.io.FixedLenFeature(
          [
              10,
          ],
          tf.int64,
          default_value=None,
      ),
      'traffic_light_state/future/state': tf.io.FixedLenFeature(
          [80, num_tls], tf.int64, default_value=None
      ),
      'traffic_light_state/future/valid': tf.io.FixedLenFeature(
          [80, num_tls], tf.int64, default_value=None
      ),
      'traffic_light_state/future/id': tf.io.FixedLenFeature(
          [80, num_tls], tf.int64, default_value=None
      ),
      'traffic_light_state/future/x': tf.io.FixedLenFeature(
          [80, num_tls], tf.float32, default_value=None
      ),
      'traffic_light_state/future/y': tf.io.FixedLenFeature(
          [80, num_tls], tf.float32, default_value=None
      ),
      'traffic_light_state/future/z': tf.io.FixedLenFeature(
          [80, num_tls], tf.float32, default_value=None
      ),
      'traffic_light_state/future/timestamp_micros': tf.io.FixedLenFeature(
          [
              80,
          ],
          tf.int64,
          default_value=None,
      ),
  }
  features_description = {}
  features_description.update(roadgraph_features)
  if include_sdc_paths:
    features_description.update({
        'path_samples/xyz': tf.io.FixedLenFeature(
            [num_paths, num_points_per_path, 3], tf.float32, default_value=None
        ),
        'path_samples/valid': tf.io.FixedLenFeature(
            [num_paths, num_points_per_path], tf.int64, default_value=None
        ),
        'path_samples/id': tf.io.FixedLenFeature(
            [num_paths, num_points_per_path], tf.int64, default_value=None
        ),
        'path_samples/arc_length': tf.io.FixedLenFeature(
            [num_paths, num_points_per_path], tf.float32, default_value=None
        ),
        'path_samples/on_route': tf.io.FixedLenFeature(
            [num_paths, 1], tf.int64, default_value=None
        ),
    })
  features_description.update(state_features)
  features_description.update(traffic_light_features)
  return features_description


def simulator_state_to_womd_dict_tensorflow(
    state: simulator_state.SimulatorState,
    feature_description: dict[str, tf.io.FixedLenFeature],
    validate: bool = False,
) -> dict[str, tf.Tensor]:
  """Tensorflow version of the simulator state to WOMD dict converter."""
  jax_dict = simulator_state_to_womd_dict(state, feature_description, validate)
  return tree.map_structure(tf.convert_to_tensor, jax_dict)


def simulator_state_to_womd_dict(
    state: simulator_state.SimulatorState,
    feature_description: dict[str, tf.io.FixedLenFeature],
    validate: bool = False,
) -> dict[str, jax.Array]:
  """Converts a simulator state into the WOMD tensor format.

  See https://waymo.com/open/data/motion/tfexample for the tf.Example format
  which will be returned from this function.
  Note: This function is compatible with `jax2tf`.

  Args:
    state: State of the simulator from the environment. Should contain at least
      num_history + 1 elements in the time dimension for all temporal components
      of the simulated trajectory.
    feature_description: Feature description expected out of the dictionary.
      This is used to understand the shape of the fields expected such as number
      of agents and amount of history.
    validate: Validate whether the simulation has progressed far enough to
      ensure that an adequate amount of history is present.

  Returns:
    A dictionary matching fields as if they were read from the WOMD dataset.

  Raises:
    ValueError: If `validate` is set to `True` and the number of history stored
      in the observations is not `num_history` + 1.
  """
  num_history = feature_description['state/past/bbox_yaw'].shape[-1]
  num_timesteps = jnp.array(state.timestep, int) + 1
  # Use checkify here so that it can be jit compilable.
  if validate:
    checkify.check(
        num_timesteps == num_history + 1,
        (
            f'Number of observations passed {num_timesteps} does not match '
            f'the expected value {num_history + 1}.'
        ),
    )

  # Agent state information.
  current_state = state.current_sim_trajectory
  state_past = operations.dynamic_slice(
      state.sim_trajectory,
      start_index=jnp.array(state.timestep, int) - num_history,
      slice_size=num_history,
      axis=-1,
  )

  # Here is the traffic light information.
  current_tl = operations.dynamic_slice(
      state.log_traffic_light, jnp.array(state.timestep, int), 1, axis=-1
  )
  tl_past = operations.dynamic_slice(
      state.log_traffic_light,
      start_index=jnp.array(state.timestep, int) - num_history,
      slice_size=num_history,
      axis=-1,
  )

  values = {}
  values.update(_roadgraph_to_dict(state.roadgraph_points, 'roadgraph_samples'))
  values.update(_trajectory_to_dict(current_state, 'current'))
  values.update(_trajectory_to_dict(state_past, 'past'))
  values.update(
      _traffic_light_to_dict(
          current_tl, 'current', current_state.timestamp_micros[..., 0, :]
      )
  )
  values.update(
      _traffic_light_to_dict(
          tl_past, 'past', state_past.timestamp_micros[..., 0, :]
      )
  )
  values.update(_object_metadata_to_dict(state.object_metadata))
  values.update(_get_invalid_future_trajectory(feature_description))
  values.update(_get_invalid_future_traffic_light(feature_description))
  # Check that all of the values conform to the feature_description and only
  # return those that are requested by the feature description.
  requested_values = {}
  for name, spec in feature_description.items():
    requested_values[name] = jnp.reshape(values[name], spec.shape).astype(
        _TF_TO_JNP_DTYPE[spec.dtype]
    )
  return requested_values


def _roadgraph_to_dict(
    rg: roadgraph.RoadgraphPoints, prefix: str = 'roadgraph_samples'
) -> dict[str, jax.Array]:
  """Gets the roadgrpah mpdata from the simulator state."""
  return {
      f'{prefix}/dir': rg.dir_xyz,
      f'{prefix}/id': rg.ids,
      f'{prefix}/type': rg.types,
      f'{prefix}/valid': rg.valid,
      f'{prefix}/xyz': rg.xyz,
  }


def _trajectory_to_dict(
    trajectory: object_state.Trajectory, time_prefix: str
) -> dict[str, jax.Array]:
  """Generates the mpdata fields for Trajectory data."""
  return {
      f'state/{time_prefix}/bbox_yaw': trajectory.yaw,
      f'state/{time_prefix}/height': trajectory.height,
      f'state/{time_prefix}/length': trajectory.length,
      f'state/{time_prefix}/speed': trajectory.speed,
      f'state/{time_prefix}/timestamp_micros': trajectory.timestamp_micros,
      f'state/{time_prefix}/valid': trajectory.valid,
      f'state/{time_prefix}/vel_yaw': trajectory.vel_yaw,
      f'state/{time_prefix}/velocity_x': trajectory.vel_x,
      f'state/{time_prefix}/velocity_y': trajectory.vel_y,
      f'state/{time_prefix}/width': trajectory.width,
      f'state/{time_prefix}/x': trajectory.x,
      f'state/{time_prefix}/y': trajectory.y,
      f'state/{time_prefix}/z': trajectory.z,
  }


def _traffic_light_to_dict(
    tls: traffic_lights.TrafficLights,
    time_prefix: str,
    timestamp_micros: jax.Array,
) -> dict[str, jax.Array]:
  """Generates the corresponding mpdata for TrafficLights fields."""
  return {
      f'traffic_light_state/{time_prefix}/state': jnp.swapaxes(tls.state, 0, 1),
      f'traffic_light_state/{time_prefix}/valid': jnp.swapaxes(tls.valid, 0, 1),
      f'traffic_light_state/{time_prefix}/x': jnp.swapaxes(tls.x, 0, 1),
      f'traffic_light_state/{time_prefix}/y': jnp.swapaxes(tls.y, 0, 1),
      f'traffic_light_state/{time_prefix}/z': jnp.swapaxes(tls.z, 0, 1),
      f'traffic_light_state/{time_prefix}/id': jnp.swapaxes(tls.lane_ids, 0, 1),
      f'traffic_light_state/{time_prefix}/timestamp_micros': timestamp_micros,
  }


def _object_metadata_to_dict(
    metadata: object_state.ObjectMetadata,
) -> dict[str, jax.Array]:
  """Converts object metadata to the original tf.Example format."""
  metadata_dict = {
      'state/type': metadata.object_types,
      'state/id': metadata.ids,
      'state/objects_of_interest': metadata.objects_of_interest,
      'state/is_sdc': metadata.is_sdc,
      'state/tracks_to_predict': metadata.is_modeled,
  }

  def _set_invalid(x: jax.Array) -> jax.Array:
    x = x.astype(jnp.int32)
    return jnp.where(metadata.is_valid, x, jnp.ones_like(x) * DEFAULT_INT)

  return tree.map_structure(_set_invalid, metadata_dict)


def _get_invalid_future_trajectory(
    feature_description: dict[str, tf.io.FixedLenFeature]
) -> dict[str, jax.Array]:
  """Gets an invalid trajectory representing future."""
  traj = object_state.Trajectory.zeros(
      feature_description['state/future/valid'].shape
  )
  traj = object_state.fill_invalid_trajectory(traj)
  traj_dict = _trajectory_to_dict(traj, 'future')
  traj_dict['state/future/speed'] = traj.speed
  traj_dict['state/future/vel_yaw'] = traj.vel_yaw
  traj_dict['state/future/valid'] = (
      jnp.ones(traj.valid.shape, jnp.int32) * DEFAULT_INT
  )
  return traj_dict


def _get_invalid_future_traffic_light(
    feature_description: dict[str, tf.io.FixedLenFeature]
) -> dict[str, jax.Array]:
  """Gets an invalid traffic light representing future."""

  def _ones_like(field_name: str) -> jax.Array:
    spec = feature_description[field_name]
    tensor = jnp.ones(spec.shape, _TF_TO_JNP_DTYPE[spec.dtype])
    if tensor.dtype in [tf.int32, tf.int64]:
      return tensor * DEFAULT_INT
    elif tensor.dtype in [tf.float32, tf.float64]:
      return tensor * DEFAULT_FLOAT
    else:
      raise ValueError(f'{tensor.dtype} not supported for {field_name}.')

  invalid_future_traffic_light = {}
  for name in feature_description:
    if 'traffic_light_state/future' not in name:
      continue
    invalid_future_traffic_light[name] = _ones_like(name)
  return invalid_future_traffic_light
