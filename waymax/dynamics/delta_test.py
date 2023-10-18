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

import jax
import jax.numpy as jnp
import tensorflow as tf

from absl.testing import parameterized
from waymax import config as _config
from waymax import dataloader
from waymax import datatypes
from waymax.dynamics import delta
from waymax.utils import geometry
from waymax.utils import test_utils

TEST_DATA_PATH = test_utils.ROUTE_DATA_PATH


class DynamicsTest(tf.test.TestCase, parameterized.TestCase):
  """Tests the action specs in the new random environment."""

  def setUp(self):
    super().setUp()
    self.sim_state = next(
        dataloader.simulator_state_generator(
            config=_config.DatasetConfig(
                path=TEST_DATA_PATH,
                max_num_objects=32,
                data_format=_config.DataFormat.TFRECORD,
            )
        )
    )

  @parameterized.product(
      batch_dims=((), (2, 1)),
      which_model=('DeltaGlobal', 'DeltaLocal'),
      controlled_obj=(
          _config.ObjectType.SDC,
          _config.ObjectType.MODELED,
      ),
  )
  def test_dynamics_model_recovers_log_by_inverse_and_forward(
      self, batch_dims, which_model, controlled_obj
  ):
    # This test checks that forward and inverse are inverses of each other.
    # We first infer an action using inverse dynamics, and run the forward
    # update. We check that the forward update matches the original trajectory.
    # Read the initial state at t=0 from the dataset.
    state_t0 = next(
        dataloader.simulator_state_generator(
            config=_config.DatasetConfig(
                path=TEST_DATA_PATH,
                max_num_objects=32,
                batch_dims=batch_dims,
                data_format=_config.DataFormat.TFRECORD,
            )
        )
    )
    # Initialize the first 10 steps of sim_trajectory with log_trajectory.
    state_t10 = datatypes.update_state_by_log(state_t0, num_steps=10)
    # This test expects all objects to be controlled.
    is_controlled = jnp.ones_like(
        datatypes.get_control_mask(state_t0.object_metadata, controlled_obj),
        dtype=jnp.bool_,
    )

    dynamics_model = getattr(delta, which_model)()
    # Run inverse dynamics on logged trajectory to infer actions.
    actions = dynamics_model.inverse(
        state_t10.log_trajectory, state_t0.object_metadata, timestep=10
    )
    # Apply inferred actions to sim trajectory to recover logged trajectory.
    out = dynamics_model.forward(
        actions,
        trajectory=state_t10.sim_trajectory,
        reference_trajectory=state_t10.log_trajectory,
        is_controlled=is_controlled,
        timestep=10,
    )
    self.assertTupleEqual(actions.shape, batch_dims + (state_t0.num_objects,))
    self.assertAllClose(out.x[..., :11], state_t10.log_trajectory.x[..., :11])
    self.assertAllClose(
        out.x[..., 12:], jnp.zeros_like(out.x[..., 12:], dtype=jnp.float32)
    )
    # Recovers for valid (both controlled and un-controlled).
    self.assertAllClose(
        out.x[actions.valid[..., 0], 11],
        state_t10.log_trajectory.x[actions.valid[..., 0], 11],
    )
    self.assertAllClose(
        out.y[actions.valid[..., 0], 11],
        state_t10.log_trajectory.y[actions.valid[..., 0], 11],
    )
    self.assertAllClose(
        out.yaw[actions.valid[..., 0], 11],
        state_t10.log_trajectory.yaw[actions.valid[..., 0], 11],
    )

  @parameterized.parameters('DeltaGlobal', 'DeltaLocal')
  def test_dynamics_model_is_jitable(self, which_model):
    is_controlled = datatypes.get_control_mask(
        self.sim_state.object_metadata, _config.ObjectType.MODELED
    )
    dynamics_model = getattr(delta, which_model)()
    actions = jax.jit(dynamics_model.inverse)(
        self.sim_state.log_trajectory,
        metadata=self.sim_state.object_metadata,
        timestep=10,
    )
    jax.jit(dynamics_model.forward)(
        actions,
        trajectory=self.sim_state.sim_trajectory,
        reference_trajectory=self.sim_state.log_trajectory,
        is_controlled=is_controlled,
        timestep=10,
    )

  @parameterized.parameters(((),), ((2, 1),))
  def test_global_dynamics_computes_accurate_actions(self, batch_dims):
    sim_state = next(
        dataloader.simulator_state_generator(
            config=_config.DatasetConfig(
                path=TEST_DATA_PATH,
                max_num_objects=32,
                batch_dims=batch_dims,
                data_format=_config.DataFormat.TFRECORD,
            )
        )
    )
    log_traj = sim_state.log_trajectory
    dynamics_model = delta.DeltaGlobal()

    # Note there are objects valid at timestep 18 but invalid at timestep 19
    # in this example.
    t = 18
    actions = dynamics_model.inverse(
        log_traj, sim_state.object_metadata, timestep=t
    )

    valid = jnp.logical_and(log_traj.valid[..., t], log_traj.valid[..., t + 1])
    raw_dx_dy = log_traj.xy[..., t + 1, :] - log_traj.xy[..., t, :]
    raw_dyaw = log_traj.yaw[..., t + 1, None] - log_traj.yaw[..., t, None]
    raw_dyaw = geometry.wrap_yaws(raw_dyaw)
    raw = jnp.concatenate([raw_dx_dy, raw_dyaw], axis=-1)
    expect = jnp.where(valid[..., None], raw, 0.0)

    self.assertAllClose(actions.data, expect)
    self.assertAllClose(actions.valid, valid[..., None])

  def test_delta_local_computes_accurate_actions(self):
    log_traj = self.sim_state.log_trajectory
    dynamics_model = delta.DeltaLocal()

    # Note there are objects valid at timestep 18 but invalid at timestep 19
    # in this example.
    t = 18
    actions = dynamics_model.inverse(
        log_traj, self.sim_state.object_metadata, timestep=t
    )

    valid = jnp.logical_and(log_traj.valid[..., t], log_traj.valid[..., t + 1])
    raw_dx_dy = log_traj.xy[..., t + 1, :] - log_traj.xy[..., t, :]
    rotation_matrix = geometry.rotation_matrix_2d(-log_traj.yaw[..., t])
    local_dx_dy = geometry.rotate_points(rotation_matrix, raw_dx_dy)
    raw_dyaw = log_traj.yaw[..., t + 1, None] - log_traj.yaw[..., t, None]
    raw_dyaw = geometry.wrap_yaws(raw_dyaw)
    raw = jnp.concatenate([local_dx_dy, raw_dyaw], axis=-1)
    expect = jnp.where(valid[..., None], raw, 0.0)

    self.assertAllClose(actions.data, expect)
    self.assertAllClose(actions.valid, valid[..., None])


if __name__ == '__main__':
  tf.test.main()
