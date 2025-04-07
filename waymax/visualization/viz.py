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

"""Visualization functions for Waymax data structures."""

from typing import Any, Optional

import jax
import matplotlib
import numpy as np

from waymax import config as waymax_config
from waymax import datatypes
from waymax.env.wrappers import brax_wrapper
from waymax.utils import geometry
from waymax.visualization import color
from waymax.visualization import utils

# The type ids for road graph elements that will be plot in visualization.
# Please refer to color.py for definition and color associcated.
_RoadGraphShown = (1, 2, 3, 15, 16, 17, 18, 19)
_RoadGraphDefaultColor = (0.9, 0.9, 0.9)


def _plot_bounding_boxes(
    ax: matplotlib.axes.Axes,
    traj_5dof: np.ndarray,
    time_idx: int,
    is_controlled: np.ndarray,
    valid: np.ndarray,
    add_label: bool = False,
) -> None:
  """Helper function to plot multiple bounding boxes across time."""
  # Plots bounding boxes (traj_5dof) with shape: (A, T)
  # is_controlled: (A,)
  # valid: (A, T)
  valid_controlled = is_controlled[:, np.newaxis] & valid
  valid_context = ~is_controlled[:, np.newaxis] & valid

  num_obj = traj_5dof.shape[0]
  time_indices = np.tile(
      np.arange(traj_5dof.shape[1])[np.newaxis, :], (num_obj, 1)
  )
  # Shrinks bounding_boxes for non-current steps.
  traj_5dof[time_indices != time_idx, 2:4] /= 10
  utils.plot_numpy_bounding_boxes(
      ax=ax,
      bboxes=traj_5dof[(time_indices >= time_idx) & valid_controlled],
      color=color.COLOR_DICT['controlled'],
      label='controlled' if add_label else None,
  )

  utils.plot_numpy_bounding_boxes(
      ax=ax,
      bboxes=traj_5dof[(time_indices < time_idx) & valid],
      color=color.COLOR_DICT['history'],
      as_center_pts=True,
      label='history' if add_label else None,
  )

  utils.plot_numpy_bounding_boxes(
      ax=ax,
      bboxes=traj_5dof[(time_indices >= time_idx) & valid_context],
      color=color.COLOR_DICT['context'],
      label='context' if add_label else None,
  )

  # Shows current overlap
  # (A, A)
  overlap_fn = jax.jit(geometry.compute_pairwise_overlaps)
  overlap_mask_matrix = overlap_fn(traj_5dof[:, time_idx])
  # Remove overlap against invalid objects.
  overlap_mask_matrix = np.where(
      valid[None, :, time_idx], overlap_mask_matrix, False
  )
  # (A,)
  overlap_mask = np.any(overlap_mask_matrix, axis=1)

  utils.plot_numpy_bounding_boxes(
      ax=ax,
      bboxes=traj_5dof[:, time_idx][overlap_mask & valid[:, time_idx]],
      color=color.COLOR_DICT['overlap'],
      label='overlap' if add_label else None,
  )


def _index_pytree(inputs: Any, idx: int) -> Any:
  """Helper function to get idx-th example in a batch."""

  def local_index(x):
    if x.ndim > 0:
      return x[idx]
    else:
      return x

  return jax.tree_util.tree_map(local_index, inputs)


def plot_trajectory(
    ax: matplotlib.axes.Axes,
    traj: datatypes.Trajectory,
    is_controlled: np.ndarray,
    time_idx: Optional[int] = None,
    indices: Optional[np.ndarray] = None,
    add_label: bool = False,
) -> None:
  """Plots a Trajectory with different color for controlled and context.

  Plots the full bounding_boxes only for time_idx step, overlap is
  highlighted.

  Notation: A: number of agents; T: number of time steps; 5 degree of freedom:
  center x, center y, length, width, yaw.

  Args:
    ax: matplotlib axes.
    traj: a Trajectory with shape (A, T).
    is_controlled: binary mask for controlled object, shape (A,).
    time_idx: step index to highlight bbox, -1 for last step. Default(None) for
      not showing bbox.
    indices: ids to show for each agents if not None, shape (A,).
    add_label: a boolean that indicates whether or not to plot labels that
      indicates different agent types, including 'controlled', 'overlap',
      'history', 'context'.
  """
  if len(traj.shape) != 2:
    raise ValueError('traj should have shape (A, T)')

  traj_5dof = np.array(
      traj.stack_fields(['x', 'y', 'length', 'width', 'yaw'])
  )  # Forces to np from jnp

  num_obj, num_steps, _ = traj_5dof.shape
  if time_idx is not None:
    if time_idx == -1:
      time_idx = num_steps - 1
    if time_idx >= num_steps:
      raise ValueError('time_idx is out of range.')

  # Adds id if needed.
  if indices is not None and time_idx is not None:
    for i in range(num_obj):
      if not traj.valid[i, time_idx]:
        continue
      ax.text(
          traj_5dof[i, time_idx, 0] - 2,
          traj_5dof[i, time_idx, 1] + 2,
          f'{indices[i]}',
          zorder=10,
      )
  _plot_bounding_boxes(
      ax=ax,
      traj_5dof=traj_5dof,
      time_idx=time_idx,
      is_controlled=is_controlled,
      valid=traj.valid,
      add_label=add_label,
  )  # pytype: disable=wrong-arg-types  # jax-ndarray


def plot_roadgraph_points(
    ax: matplotlib.axes.Axes,
    rg_pts: datatypes.RoadgraphPoints,
    verbose: bool = False,
) -> None:
  """Plots road graph as points.

  Args:
    ax: matplotlib axes.
    rg_pts: a RoadgraphPoints with shape (1,)
    verbose: print roadgraph points count if set to True.
  """
  if len(rg_pts.shape) != 1:
    raise ValueError(f'Roadgraph should be rank 1, got {len(rg_pts.shape)}')
  if rg_pts.valid.sum() == 0:
    return
  elif verbose:
    print(f'Roadgraph points count: {rg_pts.valid.sum()}')

  xy = rg_pts.xy[rg_pts.valid]
  rg_type = rg_pts.types[rg_pts.valid]
  for curr_type in np.unique(rg_type):
    if curr_type in _RoadGraphShown:
      p1 = xy[rg_type == curr_type]
      rg_color = color.ROAD_GRAPH_COLORS.get(curr_type, _RoadGraphDefaultColor)
      ax.plot(p1[:, 0], p1[:, 1], '.', color=rg_color, ms=2)


def plot_traffic_light_signals_as_points(
    ax: matplotlib.axes.Axes,
    tls: datatypes.TrafficLights,
    timestep: int = 0,
    verbose: bool = False,
) -> None:
  """Plots traffic lights for timestep.

  Args:
    ax: matplotlib axes.
    tls: a TrafficLightStates to show.
    timestep: draw traffi lights at this given timestep.
    verbose: print traffic lights count if set to True.
  """
  if len(tls.shape) != 2:
    raise ValueError('Traffic light shape wrong.')

  valid = tls.valid[:, timestep]
  if valid.sum() == 0:
    return
  elif verbose:
    print(f'Traffic lights count: {valid.sum()}')

  tls_xy = tls.xy[:, timestep][valid]
  tls_state = tls.state[:, timestep][valid]

  for xy, state in zip(tls_xy, tls_state):
    tl_color = color.TRAFFIC_LIGHT_COLORS[int(state)]
    ax.plot(xy[0], xy[1], marker='o', color=tl_color, ms=4)


def _plot_path_points(ax: matplotlib.axes.Axes, paths: datatypes.Paths) -> None:
  """Plots on/off route paths."""
  if len(paths.shape) != 2:
    raise ValueError(f'paths rank should be 2, got shape {paths.shape}.')
  for i in range(paths.shape[0]):
    if not paths.valid[i][0]:
      continue
    xy = paths.xy[i][paths.valid[i]]
    if paths.on_route[i]:
      ax.plot(xy[:, 0], xy[:, 1], 'co', ms=5, alpha=0.1)
    else:
      ax.plot(xy[:, 0], xy[:, 1], 'go', ms=5, alpha=0.1)


def plot_simulator_state(
    state: datatypes.SimulatorState,
    use_log_traj: bool = True,
    viz_config: Optional[dict[str, Any]] = None,
    batch_idx: int = -1,
    highlight_obj: waymax_config.ObjectType = waymax_config.ObjectType.SDC,
) -> np.ndarray:
  """Plots np array image for SimulatorState.

  Args:
    state: A SimulatorState instance.
    use_log_traj: Set True to use logged trajectory, o/w uses simulated
      trajectory.
    viz_config: dict for optional config.
    batch_idx: optional batch index.
    highlight_obj: Represents the type of objects that will be highlighted with
      `color.COLOR_DICT['controlled']` color.

  Returns:
    np image.
  """
  if batch_idx > -1:
    if len(state.shape) != 1:
      raise ValueError(
          'Expecting one batch dimension, got %s' % len(state.shape)
      )
    state = _index_pytree(state, batch_idx)
  if state.shape:
    raise ValueError('Expecting 0 batch dimension, got %s' % len(state.shape))

  viz_config = (
      utils.VizConfig() if viz_config is None else utils.VizConfig(**viz_config)
  )
  fig, ax = utils.init_fig_ax(viz_config)

  # 1. Plots trajectory.
  traj = state.log_trajectory if use_log_traj else state.sim_trajectory
  indices = np.arange(traj.num_objects) if viz_config.show_agent_id else None
  is_controlled = datatypes.get_control_mask(
      state.object_metadata, highlight_obj
  )
  plot_trajectory(
      ax, traj, is_controlled, time_idx=state.timestep, indices=indices
  )  # pytype: disable=wrong-arg-types  # jax-ndarray

  # 2. Plots road graph elements.
  plot_roadgraph_points(ax, state.roadgraph_points, verbose=False)
  plot_traffic_light_signals_as_points(
      ax, state.log_traffic_light, state.timestep, verbose=False
  )

  # 3. Gets np img, centered on selected agent's current location.
  # [A, 2]
  current_xy = traj.xy[:, state.timestep, :]
  if viz_config.center_agent_idx == -1:
    xy = current_xy[state.object_metadata.is_sdc]
  else:
    xy = current_xy[viz_config.center_agent_idx]
  origin_x, origin_y = xy[0, :2]
  ax.axis((
      origin_x - viz_config.back_x,
      origin_x + viz_config.front_x,
      origin_y - viz_config.back_y,
      origin_y + viz_config.front_y,
  ))

  return utils.img_from_fig(fig)


def plot_observation(
    obs: datatypes.Observation,
    obj_idx: int,
    viz_config: Optional[dict[str, Any]] = None,
    batch_idx: int = -1,
    highlight_obj: waymax_config.ObjectType = waymax_config.ObjectType.SDC,
) -> np.ndarray:
  """Plots np array image for an Observation.

  Args:
    obs: An Observation instance, with shape (..., obs_A), where obs_A
      represents the number of objects that have observation view over things
      including other objects, roadgraph, and traffic lights.
    obj_idx: The object index in obs_A.
    viz_config: Dict for optional config.
    batch_idx: Optional batch index.
    highlight_obj: Represents the type of objects that will be highlighted with
      `color.COLOR_DICT['controlled']` color.

  Returns:
    np image.
  """
  if batch_idx > -1:
    if len(obs.shape) != 2:
      raise ValueError(f'Expecting ndim 2 for obs, got {len(obs.shape)}')
    obs = jax.tree_util.tree_map(lambda x: x[batch_idx], obs)

  # Shape: (obs_A,) -> ()
  obs = jax.tree.map(lambda x: x[obj_idx], obs)
  if obs.shape:
    raise ValueError(f'Expecting shape () for obs, got {obs.shape}')

  viz_config = (
      utils.VizConfig() if viz_config is None else utils.VizConfig(**viz_config)
  )
  fig, ax = utils.init_fig_ax(viz_config)

  # 1. Plots trajectory.
  # Shape: (num_objects, num_timesteps).
  traj = obs.trajectory
  # The current timestep index of observation is the last one in time dimension
  # as observation is toward the past.
  timestep = traj.num_timesteps - 1
  indices = np.arange(traj.num_objects) if viz_config.show_agent_id else None

  is_controlled = datatypes.get_control_mask(obs.metadata, highlight_obj)
  plot_trajectory(ax, traj, is_controlled, time_idx=timestep, indices=indices)  # pytype: disable=wrong-arg-types  # jax-ndarray

  # 2. Plots road graph elements.
  # Shape: (num_points,)
  plot_roadgraph_points(ax, obs.roadgraph_static_points, verbose=False)

  # Shape: (num_traffic_lights, num_timesteps).
  plot_traffic_light_signals_as_points(
      ax, obs.traffic_lights, timestep, verbose=False
  )

  # 3. Gets np img, centered on selected agent's current location.
  # Shape (num_objects, 2).
  current_xy = traj.xy[:, timestep, :]
  if viz_config.center_agent_idx == -1:
    xy = current_xy[obs.metadata.is_sdc]
  else:
    xy = current_xy[viz_config.center_agent_idx]
  origin_x, origin_y = xy[0, :2]
  ax.axis((
      origin_x - viz_config.back_x,
      origin_x + viz_config.front_x,
      origin_y - viz_config.back_y,
      origin_y + viz_config.front_y,
  ))

  return utils.img_from_fig(fig)


def plot_single_agent_brax_timestep(
    waymax_ts: brax_wrapper.TimeStep,
    use_log_traj: bool = False,
    viz_config: Optional[dict[str, Any]] = None,
    batch_idx: int = -1,
) -> np.ndarray:
  """Plots np array image for Brax TimeStep with metrics.

  Currently only for single-agent env outputs.

  Args:
    waymax_ts: Timestep returned from Waymax env step or reset.
    use_log_traj: Set True to use logged trajectory, o/w uses simulated
      trajectory.
    viz_config: dict for optional config.
    batch_idx: optional batch index.

  Returns:
    np image.
  """
  state = waymax_ts.state
  metric_dict = waymax_ts.metrics

  if batch_idx > -1:
    if len(state.shape) != 1:
      raise ValueError(
          'Expecting one batch dimension, got %s' % len(state.shape)
      )
    state = _index_pytree(state, batch_idx)
    metric_dict = _index_pytree(metric_dict, batch_idx)
  if state.shape:
    raise ValueError('Expecting 0 batch dimension, got %s' % len(state.shape))

  viz_config = (
      utils.VizConfig() if viz_config is None else utils.VizConfig(**viz_config)
  )
  fig, ax = utils.init_fig_ax(viz_config)

  # 1. Plots trajectory.
  traj = state.log_trajectory if use_log_traj else state.sim_trajectory
  indices = np.arange(traj.num_objects) if viz_config.show_agent_id else None
  plot_trajectory(
      ax,
      traj,
      state.object_metadata.is_sdc,
      time_idx=state.timestep,
      indices=indices,
  )

  # 2. Plots road graph elements.
  plot_roadgraph_points(ax, state.roadgraph_points, verbose=False)
  plot_traffic_light_signals_as_points(
      ax, state.log_traffic_light, state.timestep, verbose=False
  )

  # Plots paths.
  _plot_path_points(ax, state.sdc_paths)

  # 3. Gets np img, centered on selected agent's current location.
  # [A, 2]
  current_xy = traj.xy[:, state.timestep, :]
  if viz_config.center_agent_idx == -1:
    xy = current_xy[state.object_metadata.is_sdc]
  else:
    xy = current_xy[viz_config.center_agent_idx]
  origin_x, origin_y = xy[0, :2]
  ax.axis((
      origin_x - viz_config.back_x,
      origin_x + viz_config.front_x,
      origin_y - viz_config.back_y,
      origin_y + viz_config.front_y,
  ))

  # 4. Added text for metric_dict.
  for metric_name, metric_result in metric_dict.items():
    if metric_result.valid:
      metric_info = {metric_name: f'{float(metric_result.value):.2f}'}
      ax.text(
          origin_x - viz_config.back_x,
          origin_y - viz_config.back_y + 10,
          metric_info,
          fontsize=6,
      )

  return utils.img_from_fig(fig)
