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

"""Align logged traffic-light states to agent trajectory timestamps.

WOMD stores per-step traffic-light timestamps separately from agent timestamps.
In ~10% of scenarios (waymax#60) the indices match but timestamps differ by one
or more 100 ms steps. Waymax previously coupled modalities by index only, which
mis-pairs TL phase with logged motion.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from waymax.datatypes import object_state
from waymax.datatypes import traffic_lights

# WOMD motion examples use 10 Hz (100 ms) sampling.
_WOMD_STEP_MICROS = 100_000


def extract_traffic_light_timestamps(
    example: dict[str, jax.Array], time_key: str = 'all'
) -> jax.Array | None:
  """Returns `(num_timesteps,)` TL timestamps or None if absent."""
  key = f'traffic_light_state/{time_key}/timestamp_micros'
  if key not in example:
    return None
  return jnp.asarray(example[key], dtype=jnp.int64)


def reference_agent_timestamps(
    trajectory: object_state.Trajectory,
    is_sdc: jax.Array | None = None,
) -> jax.Array:
  """Reference timeline `(num_timesteps,)` for alignment.

  Prefers the SDC; otherwise the first valid modeled agent.
  """
  agent_ts = trajectory.timestamp_micros
  if agent_ts.ndim == 1:
    return agent_ts.astype(jnp.int64)

  if is_sdc is not None and jnp.any(is_sdc):
    sdc_idx = jnp.argmax(is_sdc.astype(jnp.int32))
    return agent_ts[sdc_idx].astype(jnp.int64)

  valid = trajectory.valid
  if valid.ndim == 2 and jnp.any(valid):
    # First agent with any valid step.
    agent_idx = jnp.argmax(jnp.any(valid, axis=-1).astype(jnp.int32))
    return agent_ts[agent_idx].astype(jnp.int64)

  return agent_ts[0].astype(jnp.int64)


def timestamps_aligned(
    agent_ts: jax.Array, tl_ts: jax.Array, tolerance_micros: int = 1
) -> jax.Array:
  """True when every index pairs equal timestamps (within tolerance)."""
  if agent_ts.shape != tl_ts.shape:
    return jnp.array(False)
  return jnp.all(jnp.abs(agent_ts - tl_ts) <= tolerance_micros)


def estimate_index_offset(
    agent_ts: jax.Array,
    tl_ts: jax.Array,
    max_offset: int = 2,
) -> int:
  """Integer shift to apply to TL indices (positive shifts TL data forward)."""
  length = int(min(agent_ts.shape[0], tl_ts.shape[0]))
  if length == 0:
    return 0

  agent_ts = agent_ts[:length]
  tl_ts = tl_ts[:length]
  best_offset = 0
  best_matches = -1
  for offset in range(-max_offset, max_offset + 1):
    if offset >= 0:
      a = agent_ts[offset:]
      t = tl_ts[: length - offset]
    else:
      a = agent_ts[: length + offset]
      t = tl_ts[-offset:]
    if a.shape[0] == 0:
      continue
    matches = int(jnp.sum(a == t))
    if matches > best_matches:
      best_matches = matches
      best_offset = offset
  return best_offset


def _shift_along_time(
    tls: traffic_lights.TrafficLights, offset: int
) -> traffic_lights.TrafficLights:
  """Shift TL tensors along the time axis by `offset` (edge-padded)."""
  if offset == 0:
    return tls

  def shift(x: jax.Array) -> jax.Array:
    if offset > 0:
      pad = jax.lax.slice_in_dim(x, 0, 1, axis=-1)
      pad = jnp.repeat(pad, offset, axis=-1)
      return jnp.concatenate([pad, x[..., :-offset]], axis=-1)
    neg = -offset
    pad = jax.lax.slice_in_dim(x, x.shape[-1] - 1, x.shape[-1], axis=-1)
    pad = jnp.repeat(pad, neg, axis=-1)
    return jnp.concatenate([x[..., neg:], pad], axis=-1)

  return jax.tree_util.tree_map(shift, tls)


def resample_traffic_lights_nearest(
    tls: traffic_lights.TrafficLights,
    source_ts: jax.Array,
    target_ts: jax.Array,
) -> traffic_lights.TrafficLights:
  """Resample TL states so output timestep `t` uses nearest `source_ts` entry."""
  num_steps = target_ts.shape[0]
  # For each target time, index of closest source time.
  dist = jnp.abs(
      target_ts[:, jnp.newaxis].astype(jnp.int64)
      - source_ts[jnp.newaxis, :].astype(jnp.int64)
  )
  indices = jnp.argmin(dist, axis=1)

  def gather_time(x: jax.Array) -> jax.Array:
    # x: (..., num_tls, num_source_steps)
    return jnp.take(x, indices, axis=-1)

  return jax.tree_util.tree_map(gather_time, tls)


def align_to_trajectory(
    tls: traffic_lights.TrafficLights,
    trajectory: object_state.Trajectory,
    example: dict[str, jax.Array],
    time_key: str = 'all',
    is_sdc: jax.Array | None = None,
    max_offset: int = 2,
) -> traffic_lights.TrafficLights:
  """Return TL states resampled onto the agent reference timeline if needed."""
  tl_ts = extract_traffic_light_timestamps(example, time_key=time_key)
  if tl_ts is None:
    return tls

  agent_ts = reference_agent_timestamps(trajectory, is_sdc=is_sdc)
  if timestamps_aligned(agent_ts, tl_ts):
    return tls

  offset = estimate_index_offset(agent_ts, tl_ts, max_offset=max_offset)
  shifted = _shift_along_time(tls, offset)
  shifted_ts = _shift_along_time(
      traffic_lights.TrafficLights(
          x=tl_ts.astype(jnp.float32),
          y=jnp.zeros_like(tl_ts, dtype=jnp.float32),
          z=jnp.zeros_like(tl_ts, dtype=jnp.float32),
          state=jnp.zeros_like(tl_ts, dtype=jnp.int32),
          lane_ids=jnp.zeros_like(tl_ts, dtype=jnp.int32),
          valid=jnp.ones_like(tl_ts, dtype=jnp.bool_),
      ),
      offset,
  ).x.astype(jnp.int64)

  if timestamps_aligned(agent_ts, shifted_ts):
    return shifted

  return resample_traffic_lights_nearest(tls, tl_ts, agent_ts)
