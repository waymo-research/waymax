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

import jax.numpy as jnp
import tensorflow as tf

from absl.testing import absltest
from absl.testing import parameterized
from waymax.dataloader import traffic_light_alignment
from waymax.datatypes import object_state
from waymax.datatypes import traffic_lights


def _make_trajectory(
    timestamps: list[int], num_agents: int = 1
) -> object_state.Trajectory:
  ts = jnp.asarray(timestamps, dtype=jnp.int32)
  if num_agents > 1:
    ts = jnp.tile(ts[jnp.newaxis, :], (num_agents, 1))
  valid = jnp.ones(ts.shape, dtype=jnp.bool_)
  return object_state.Trajectory(
      x=jnp.zeros(ts.shape, jnp.float32),
      y=jnp.zeros(ts.shape, jnp.float32),
      z=jnp.zeros(ts.shape, jnp.float32),
      vel_x=jnp.zeros(ts.shape, jnp.float32),
      vel_y=jnp.zeros(ts.shape, jnp.float32),
      yaw=jnp.zeros(ts.shape, jnp.float32),
      valid=valid,
      length=jnp.ones(ts.shape, jnp.float32),
      width=jnp.ones(ts.shape, jnp.float32),
      height=jnp.ones(ts.shape, jnp.float32),
      timestamp_micros=ts,
  )


def _make_tls(states: list[int], num_tls: int = 1) -> traffic_lights.TrafficLights:
  state = jnp.asarray(states, dtype=jnp.int32)
  if state.ndim == 1:
    state = state[jnp.newaxis, :]
  if num_tls > 1:
    state = jnp.tile(state, (num_tls, 1))
  return traffic_lights.TrafficLights(
      x=jnp.zeros(state.shape, jnp.float32),
      y=jnp.zeros(state.shape, jnp.float32),
      z=jnp.zeros(state.shape, jnp.float32),
      state=state,
      lane_ids=jnp.ones(state.shape, jnp.int32),
      valid=jnp.ones(state.shape, jnp.bool_),
  )


class TrafficLightAlignmentTest(parameterized.TestCase, tf.test.TestCase):

  def test_timestamps_aligned_true_when_equal(self):
    ts = jnp.array([0, 100_000, 200_000], dtype=jnp.int64)
    self.assertTrue(traffic_light_alignment.timestamps_aligned(ts, ts))

  def test_estimate_offset_one_step_behind(self):
    agent = jnp.array([0, 100_000, 200_000, 300_000], dtype=jnp.int64)
    tl = jnp.array([100_000, 200_000, 300_000, 400_000], dtype=jnp.int64)
    self.assertEqual(traffic_light_alignment.estimate_index_offset(agent, tl), 1)

  def test_shift_restores_alignment(self):
    agent_ts = jnp.array([0, 100_000, 200_000, 300_000], dtype=jnp.int64)
    tl_ts = jnp.array([100_000, 200_000, 300_000, 400_000], dtype=jnp.int64)
    tls = _make_tls([4, 6, 6, 6])
    offset = traffic_light_alignment.estimate_index_offset(agent_ts, tl_ts)
    shifted = traffic_light_alignment._shift_along_time(tls, offset)
    # TL timestamps lag by one: agent t=2 (200ms) pairs with tl index 1 (GO).
    self.assertEqual(int(shifted.state[0, 2]), 6)

  def test_align_to_trajectory_fixes_one_step_desync(self):
    timestamps = [0, 100_000, 200_000, 300_000]
    traj = _make_trajectory(timestamps)
    tls = _make_tls([4, 6, 6, 6])
    example = {
        'traffic_light_state/all/timestamp_micros': jnp.array(
            [100_000, 200_000, 300_000, 400_000], dtype=jnp.int64
        ),
    }
    aligned = traffic_light_alignment.align_to_trajectory(
        tls, traj, example, is_sdc=jnp.array([True])
    )
    # Agent t=2 (200ms) should see GO (6) from TL index 1 after shift.
    self.assertEqual(int(aligned.state[0, 2]), 6)

  def test_align_noop_when_already_aligned(self):
    timestamps = [0, 100_000, 200_000]
    traj = _make_trajectory(timestamps)
    tls = _make_tls([4, 5, 6])
    example = {
        'traffic_light_state/all/timestamp_micros': jnp.array(
            timestamps, dtype=jnp.int64
        ),
    }
    aligned = traffic_light_alignment.align_to_trajectory(
        tls, traj, example, is_sdc=jnp.array([True])
    )
    self.assertEqual(aligned, tls)

  def test_resample_nearest_handles_irregular_gap(self):
    source_ts = jnp.array([0, 250_000, 500_000], dtype=jnp.int64)
    target_ts = jnp.array([0, 100_000, 200_000], dtype=jnp.int64)
    tls = _make_tls([4, 5, 6])
    out = traffic_light_alignment.resample_traffic_lights_nearest(
        tls, source_ts, target_ts
    )
    self.assertEqual(int(out.state[0, 0]), 4)
    self.assertEqual(int(out.state[0, 1]), 4)
    self.assertEqual(int(out.state[0, 2]), 5)  # 200ms nearest to source 250ms


  def test_align_integration_via_factory(self):
    timestamps = [0, 100_000, 200_000, 300_000]
    traj = _make_trajectory(timestamps)
    tls = _make_tls([4, 6, 6, 6])
    example = {
        'state/all/x': traj.x,
        'state/all/y': traj.y,
        'state/all/z': traj.z,
        'state/all/velocity_x': traj.vel_x,
        'state/all/velocity_y': traj.vel_y,
        'state/all/bbox_yaw': traj.yaw,
        'state/all/valid': traj.valid,
        'state/all/length': traj.length,
        'state/all/width': traj.width,
        'state/all/height': traj.height,
        'state/all/timestamp_micros': traj.timestamp_micros,
        'state/is_sdc': jnp.array([1]),
        'state/id': jnp.array([1], dtype=jnp.int32),
        'state/type': jnp.array([1], dtype=jnp.int32),
        'state/tracks_to_predict': jnp.array([1], dtype=jnp.int32),
        'state/objects_of_interest': jnp.array([0], dtype=jnp.int32),
        'traffic_light_state/all/x': jnp.zeros((4, 1), jnp.float32),
        'traffic_light_state/all/y': jnp.zeros((4, 1), jnp.float32),
        'traffic_light_state/all/z': jnp.zeros((4, 1), jnp.float32),
        'traffic_light_state/all/state': jnp.transpose(tls.state),
        'traffic_light_state/all/id': jnp.ones((4, 1), jnp.int32),
        'traffic_light_state/all/valid': jnp.ones((4, 1), dtype=jnp.bool_),
        'traffic_light_state/all/timestamp_micros': jnp.array(
            [100_000, 200_000, 300_000, 400_000], dtype=jnp.int64
        ),
        'roadgraph_samples/xyz': jnp.zeros((1, 3), jnp.float32),
        'roadgraph_samples/dir': jnp.zeros((1, 3), jnp.float32),
        'roadgraph_samples/type': jnp.zeros((1, 1), jnp.int32),
        'roadgraph_samples/id': jnp.zeros((1, 1), jnp.int32),
        'roadgraph_samples/valid': jnp.ones((1, 1), dtype=jnp.bool_),
    }
    from waymax.dataloader import womd_factories

    sim = womd_factories.simulator_state_from_womd_dict(example, time_key='all')
    self.assertEqual(int(sim.log_traffic_light.state[0, 2]), 6)


if __name__ == '__main__':
  absltest.main()
