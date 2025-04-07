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
from waymax.dynamics import state_dynamics
from waymax.utils import test_utils

TEST_DATA_PATH = test_utils.ROUTE_DATA_PATH


class StateDynamicsTest(parameterized.TestCase, tf.test.TestCase):

  def test_forward_update_matches_expected_result(self):
    batch_size, objects, timesteps = 1, 5, 10
    log_traj = datatypes.Trajectory(
        x=jnp.zeros((batch_size, objects, timesteps)),
        y=jnp.zeros((batch_size, objects, timesteps)),
        z=jnp.zeros((batch_size, objects, timesteps)),
        vel_x=jnp.zeros((batch_size, objects, timesteps)),
        vel_y=jnp.zeros((batch_size, objects, timesteps)),
        yaw=jnp.zeros((batch_size, objects, timesteps)),
        valid=jnp.ones((batch_size, objects, timesteps), dtype=bool),
        timestamp_micros=jnp.zeros(
            (batch_size, objects, timesteps), dtype=jnp.int32
        ),
        length=jnp.zeros((batch_size, objects, timesteps)),
        width=jnp.zeros((batch_size, objects, timesteps)),
        height=jnp.zeros((batch_size, objects, timesteps)),
    )
    # Initialize the simulated trajectory to all one-valued attributes.
    sim_traj = jax.tree.map(jnp.ones_like, log_traj)
    # Set the 2nd object (index 1) to be controlled.
    is_controlled = jnp.array([[False, True, False, False, False]])
    # Create a test action with value (1, 2, 3, 4, 5)
    # This sets x=1, y=2, yaw=3, vel_x=4, vel_y=5
    action = jnp.tile(jnp.arange(5), (batch_size, objects, 1)) + 1
    assert action.shape == (batch_size, objects, 5)
    actions = datatypes.Action(
        data=action.astype(jnp.float32), valid=jnp.ones((1, 5, 1), dtype=bool)
    )

    dynamics_model = state_dynamics.StateDynamics()
    timestep = 2
    # Shape: (batch_size=1, objects=5, timesteps=10)
    next_traj = dynamics_model.forward(
        actions, sim_traj, log_traj, is_controlled, timestep
    )

    # Shape: (batch_size=1, objects=5, timesteps=1)
    traj_at_timestep = datatypes.dynamic_slice(
        next_traj, timestep + 1, 1, axis=-1
    )
    # Shape: (batch_size=1, timesteps=1)
    controlled_traj = jax.tree.map(lambda x: x[is_controlled], traj_at_timestep)

    with self.subTest('ControlledTrajIsCorrect'):
      self.assertAllClose(
          controlled_traj.x, 1 * tf.ones_like(controlled_traj.x)
      )
      self.assertAllClose(
          controlled_traj.y, 2 * tf.ones_like(controlled_traj.y)
      )
      self.assertAllClose(
          controlled_traj.yaw, 3 * tf.ones_like(controlled_traj.yaw)
      )
      self.assertAllClose(
          controlled_traj.vel_x, 4 * tf.ones_like(controlled_traj.vel_x)
      )
      self.assertAllClose(
          controlled_traj.vel_y, 5 * tf.ones_like(controlled_traj.vel_y)
      )

  @parameterized.product(
      batch_dims=((), (2, 1)),
      controlled_obj=(
          _config.ObjectType.SDC,
          _config.ObjectType.MODELED,
      ),
  )
  def test_dynamics_model_recovers_log_by_inverse_and_forward(
      self, batch_dims, controlled_obj
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
                data_format=_config.DataFormat.TFRECORD,
                batch_dims=batch_dims,
            )
        )
    )
    # Initialize the first 10 steps of sim_trajectory with log_trajectory.
    state_t10 = datatypes.update_state_by_log(state_t0, num_steps=10)
    is_controlled = datatypes.get_control_mask(
        state_t0.object_metadata, controlled_obj
    )

    dynamics_model = state_dynamics.StateDynamics()
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


if __name__ == '__main__':
  tf.test.main()
