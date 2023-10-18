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

"""Helper functions and utilities to provide test states for Waymax."""

from collections.abc import Sequence
import functools
import os

import jax
import jax.numpy as jnp
import tensorflow as tf

from waymax import config as _config
from waymax import datatypes
from waymax.dataloader import dataloader_utils
from waymax.dataloader import womd_dataloader
from waymax.dataloader import womd_factories
from waymax.dataloader import womd_utils
from waymax.datatypes import simulator_state

ROUTE_DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    'dataloader',
    'testdata',
    'tfrecord_with_routes',
)
ROUTE_NUM_PATHS = 30
ROUTE_NUM_POINTS_PER_PATH = 200


def make_test_dataset(batch_dims: tuple[int, ...] = (), **kwargs):
  """Create a test dataset using ROUTE_DATA_PATH."""
  config = _config.DatasetConfig(
      path=ROUTE_DATA_PATH,
      data_format=_config.DataFormat.TFRECORD,
      batch_dims=batch_dims,
      **kwargs,
  )
  preprocess_fn = functools.partial(
      womd_dataloader.preprocess_serialized_womd_data, config=config
  )
  return dataloader_utils.tf_examples_dataset(
      path=config.path,
      data_format=_config.DataFormat.TFRECORD,
      preprocess_fn=preprocess_fn,
      batch_dims=batch_dims,
  )


def simulated_trajectory_no_overlap() -> datatypes.Trajectory:
  """Creates a simulated trajectory where there are no object overlaps."""
  # Create 6 boxes away from the SDC which should all result in no overlap.
  # The agent is stored at the origin (0, 0)
  xyz = jnp.array([
      [[0.0, 0.0, 0.0]],
      [[0.0, 2.0, 0.0]],
      [[-3.0, 1.0, 0.0]],
      [[3.0, 1.0, 0.0]],
      [[3.0, -2.0, 0.0]],
      [[-2.0, -1.5, 0.0]],
      [[0.0, -3.0, 0.0]],
  ])
  num_agents = xyz.shape[0]
  yaw = jnp.arange(num_agents)[:, jnp.newaxis] * 0.5
  valid = jnp.array([[True]] * num_agents)
  width = jnp.ones((num_agents, 1), dtype=jnp.float32)
  length = jnp.ones((num_agents, 1), dtype=jnp.float32) * 2
  height = jnp.ones((num_agents, 1), dtype=jnp.float32)
  traj = datatypes.Trajectory(
      x=xyz[..., 0],
      y=xyz[..., 1],
      z=xyz[..., 2],
      vel_x=xyz[..., 0] * 0.0,
      vel_y=xyz[..., 1] * 0.0,
      yaw=yaw,
      valid=valid,
      timestamp_micros=jnp.zeros((num_agents, 1), dtype=jnp.int32),
      length=length,
      width=width,
      height=height,
  )
  traj.validate()
  return traj


def simulated_trajectory_with_overlap() -> datatypes.Trajectory:
  """Creates a simulated trajectory where the SDC has overlaps."""
  # Create 6 boxes around the SDC which should all result in an overlap.
  # The sdc is stored at the origin (0, 0)
  xyz = jnp.array([
      [[0.0, 0.0, 0.0]],
      [[0.0, 1.0, 0.0]],
      [[-1.0, 1.0, 0.0]],
      [[1.0, 1.0, 0.0]],
      [[1.0, -1.0, 0.0]],
      [[-1.0, -1.0, 0.0]],
      [[0.0, -1.0, 0.0]],
  ])
  num_agents = xyz.shape[0]
  yaw = jnp.arange(num_agents)[:, jnp.newaxis] * 0.5
  width = jnp.ones((num_agents, 1), dtype=jnp.float32)
  length = jnp.ones((num_agents, 1), dtype=jnp.float32) * 2
  height = jnp.ones((num_agents, 1), dtype=jnp.float32)
  valid = jnp.array([[True]] * num_agents)
  timestamp_micros = jnp.zeros((num_agents, 1), dtype=jnp.int32)

  return datatypes.Trajectory(
      x=xyz[..., 0],
      y=xyz[..., 1],
      z=xyz[..., 2],
      vel_x=xyz[..., 0] * 0.0,
      vel_y=xyz[..., 1] * 0.0,
      yaw=yaw,
      valid=valid,
      timestamp_micros=timestamp_micros,
      width=width,
      length=length,
      height=height,
  )


def simulator_state_with_overlap() -> datatypes.SimulatorState:
  """Creates a simulator state where the SDC has overlaps."""
  traj = simulated_trajectory_with_overlap()
  roadgraph_points = create_test_map_element(
      element_type=datatypes.MapElementIds.ROAD_EDGE_BOUNDARY,
      position=[0.0, 0.0, 0.0],
      direction=[0.0, 1.0, 0.0],
      repeat=10,
  )
  # This SDC position is on the side of the boundary where it will fire the
  # offroad metric.
  tl = datatypes.TrafficLights(
      x=jnp.array([0]),
      y=jnp.array([0]),
      z=jnp.array([0]),
      state=jnp.array([0]),
      lane_ids=jnp.array([0]),
      valid=jnp.array([0]),
  )
  return datatypes.SimulatorState(
      roadgraph_points=roadgraph_points,
      sim_trajectory=traj,
      log_trajectory=traj,
      log_traffic_light=tl,
      object_metadata=create_metadata(traj.num_objects),
      timestep=jnp.array(0),
  )


def create_metadata(num_agents: int) -> datatypes.ObjectMetadata:
  """Returns a sample metadata object."""
  controlled = [False] * num_agents
  controlled[0] = True
  controlled_objects = jnp.array(controlled)
  valid = jnp.array([[True]] * num_agents)
  metadata = datatypes.ObjectMetadata(
      ids=jnp.ones((num_agents,), dtype=jnp.int32),
      object_types=jnp.ones((num_agents,), dtype=jnp.int32),
      is_sdc=controlled_objects,
      is_modeled=controlled_objects,
      is_valid=valid,
      objects_of_interest=controlled_objects,
      is_controlled=controlled_objects,
  )
  return metadata


def create_invalid_traffic_lights() -> datatypes.TrafficLights:
  """Creates invalid traffic lights with shape ()."""
  traffic_lights = datatypes.TrafficLights(
      x=jnp.array([0], dtype=jnp.float32),
      y=jnp.array([0], dtype=jnp.float32),
      z=jnp.array([0], dtype=jnp.float32),
      state=jnp.array([0], dtype=jnp.int32),
      lane_ids=jnp.array([0], dtype=jnp.int32),
      valid=jnp.array([0], dtype=jnp.bool_),
  )
  traffic_lights.validate()
  return traffic_lights


def simulator_state_with_offroad() -> datatypes.SimulatorState:
  """Creates a simulator state with the SDC off of the road."""
  roadgraph_points = create_test_map_element(
      element_type=datatypes.MapElementIds.ROAD_EDGE_BOUNDARY,
      position=[0.0, 0.0, 0.0],
      direction=[0.0, 1.0, 0.0],
      repeat=10,
  )
  # This SDC position is on the side of the boundary where it will fire the
  # offroad metric.
  traj = create_test_trajectory_from_position(position=[3.0, 3.0, 0.0])
  tl = create_invalid_traffic_lights()
  return datatypes.SimulatorState(
      roadgraph_points=roadgraph_points,
      sim_trajectory=traj,
      log_trajectory=traj,
      log_traffic_light=tl,
      object_metadata=create_metadata(traj.num_objects),
      timestep=jnp.array(0),
  )


def simulator_state_without_offroad() -> datatypes.SimulatorState:
  """Creates a simulator state with the SDC off of the road."""
  roadgraph_points = create_test_map_element(
      element_type=datatypes.MapElementIds.ROAD_EDGE_BOUNDARY,
      position=[0.0, 0.0, 0.0],
      direction=[0.0, 1.0, 0.0],
      repeat=10,
  )
  # This SDC position is on the side of the boundary where it will not fire
  # the offroad metric.
  traj = create_test_trajectory_from_position(position=[-1.2, 0.2, 0.0])
  tl = create_invalid_traffic_lights()
  return datatypes.SimulatorState(
      roadgraph_points=roadgraph_points,
      sim_trajectory=traj,
      log_trajectory=traj,
      log_traffic_light=tl,
      object_metadata=create_metadata(traj.num_objects),
      timestep=jnp.array(0),
  )


def create_test_map_element(
    element_type: datatypes.MapElementIds,
    position: Sequence[float],
    direction: Sequence[float],
    repeat: int = 1,
) -> datatypes.RoadgraphPoints:
  """Creates a test map element given specification in arguments.

  Args:
    element_type: Type of map element to create. See
      `waymax/open_motion_data/constants.py` for the list of map elements.
    position: 3D position of the map element in the global coordinate system in
      meters.
    direction: 3D direction of the map elemnt (i.e. direction of lane) in the
      global coordinate system in meters.
    repeat: How many times to repeat the map element.

  Returns:
    A map element with `repeat` number of versions that contains the fields
      specified in the arguments.

  Raises:
    ValueError: If the provided `position` does not have three elements.
    ValueError: If the provided `direction` does not have three elements.
  """
  if len(position) != 3:
    raise ValueError('Provided position input must be a 3-element sequence.')
  if len(direction) != 3:
    raise ValueError('Provided direction input must be a 3-element sequence.')
  xyz = jnp.tile(jnp.array(position), (repeat, 1))
  direction = jnp.tile(jnp.array(direction), (repeat, 1))
  types = jnp.ones((repeat,), dtype=jnp.int32) * element_type
  valid = jnp.ones((repeat), dtype=jnp.bool_)
  ids = jnp.ones((repeat,), dtype=jnp.int32)
  return datatypes.RoadgraphPoints(
      x=xyz[..., 0],
      y=xyz[..., 1],
      z=xyz[..., 2],
      dir_x=direction[..., 0],
      dir_y=direction[..., 1],
      dir_z=direction[..., 2],
      types=types,
      valid=valid,
      ids=ids,
  )


def create_test_trajectory_from_position(
    position: Sequence[float] = (1.0, 1.0, 1.0)
) -> datatypes.Trajectory:
  """Creates a trajectory with default metadata fields at a position.

  Args:
    position: Position of the trajectory point in global coordinates.

  Returns:
    Filled trajectory with a single point at `position`.

  Raises:
    If `position` does not have three elements.
  """
  xyz = jnp.array([[position]], dtype=jnp.float32)
  if xyz.shape != (1, 1, 3):
    raise ValueError('Provided position input must be a 3-element sequence.')
  vel_xy = jnp.zeros((1, 1, 2), dtype=jnp.float32)
  yaw = jnp.zeros((1, 1), dtype=jnp.float32)
  valid = jnp.ones((1, 1)).astype(jnp.bool_)
  lwh = jnp.ones((1, 1), jnp.float32)
  timestamp_micros = jnp.zeros((1, 1), dtype=jnp.int32)
  traj = datatypes.Trajectory(
      x=xyz[..., 0],
      y=xyz[..., 1],
      z=xyz[..., 2],
      vel_x=vel_xy[..., 0],
      vel_y=vel_xy[..., 1],
      yaw=yaw,
      valid=valid,
      timestamp_micros=timestamp_micros,
      width=lwh,
      length=lwh,
      height=lwh,
  )
  traj.validate()
  return traj


def make_zeros_state(
    config: _config.DatasetConfig,
) -> simulator_state.SimulatorState:
  """Returns a SimulatorState containing zeros."""
  tf_inputs = {}
  features = womd_utils.get_features_description(
      include_sdc_paths=config.include_sdc_paths,
      max_num_rg_points=config.max_num_rg_points,
      num_paths=config.num_paths,
      num_points_per_path=config.num_points_per_path,
  )

  for k in features:
    tf_inputs[k] = tf.zeros(features[k].shape, dtype=features[k].dtype)
  tf_processed = womd_dataloader.preprocess_womd_example(
      tf_inputs,
      aggregate_timesteps=config.aggregate_timesteps,
      max_num_objects=config.max_num_objects,
  )
  jnp_inputs = jax.tree_util.tree_map(jnp.asarray, tf_processed)
  # Need to set an arbitrary object as the SDC.
  jnp_inputs['state/is_sdc'] = jnp_inputs['state/is_sdc'].at[..., 0].set(1)
  state = womd_factories.simulator_state_from_womd_dict(
      jnp_inputs,
      include_sdc_paths=config.include_sdc_paths,
  )
  # Some fields will need to be set to zero as invalid shape fields will be
  # set to -1.
  zero_trajectory = state.sim_trajectory.replace(
      length=jnp.zeros_like(state.sim_trajectory.length),
      width=jnp.zeros_like(state.sim_trajectory.width),
      height=jnp.zeros_like(state.sim_trajectory.height),
      valid=jnp.ones_like(state.sim_trajectory.valid, dtype=jnp.bool_),
  )
  return state.replace(
      sim_trajectory=zero_trajectory, log_trajectory=zero_trajectory
  )
