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

from dm_env import specs
import jax
from jax import numpy as jnp
import tensorflow as tf

from absl.testing import parameterized
from waymax import config as _config
from waymax import dataloader
from waymax import datatypes
from waymax.dynamics import abstract_dynamics
from waymax.utils import test_utils

TEST_DATA_PATH = test_utils.ROUTE_DATA_PATH


class TestDynamics(abstract_dynamics.DynamicsModel):
  """Ignores actions and returns a hard-coded trajectory update at each step."""

  def __init__(self, update: datatypes.TrajectoryUpdate):
    self.update = update

  def compute_update(
      self,
      action: datatypes.Action,
      trajectory: datatypes.Trajectory,
  ) -> datatypes.TrajectoryUpdate:
    return self.update

  def inverse(
      self,
      trajectory: datatypes.Trajectory,
      metadata: datatypes.ObjectMetadata,
      timestep: int,
  ) -> datatypes.Action:
    raise NotImplementedError()

  def action_spec(self) -> specs.BoundedArray:
    raise NotImplementedError()


class AbstractDynamicsTest(tf.test.TestCase, parameterized.TestCase):

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
    sim_traj = jax.tree.map(jnp.ones_like, log_traj)
    is_controlled = jnp.array([[True, False, False, False, False]])
    update = datatypes.TrajectoryUpdate(
        x=1 * jnp.ones((batch_size, objects, 1)),
        y=2 * jnp.ones((batch_size, objects, 1)),
        yaw=3 * jnp.ones((batch_size, objects, 1)),
        vel_x=4 * jnp.ones((batch_size, objects, 1)),
        vel_y=5 * jnp.ones((batch_size, objects, 1)),
        valid=jnp.ones((batch_size, objects, 1), dtype=bool),
    )

    # Use TestDynamics, which simply sets the state to the value of the action.
    dynamics_model = TestDynamics(update)
    timestep = 2
    next_traj = dynamics_model.forward(  # pytype: disable=wrong-arg-types  # jnp-type
        action=jnp.zeros((batch_size, objects)),
        trajectory=sim_traj,
        reference_trajectory=log_traj,
        is_controlled=is_controlled,
        timestep=timestep,
    )
    # Extract the result trajectory at timestep t+1
    next_step = datatypes.dynamic_slice(next_traj, timestep + 1, 1, axis=-1)
    # Extract the log trajectory at timestep t+1
    log_t = datatypes.dynamic_slice(log_traj, timestep + 1, 1, axis=-1)
    for field in abstract_dynamics.CONTROLLABLE_FIELDS:
      with self.subTest(field):
        # Check that the controlled fields are set to the same value
        # as the update (this is the behavior of TestDynamics),
        # and non-controlled fields are set to the value
        # contained in the logs.
        self.assertAllClose(
            next_step[field][..., 0:1, :], update[field][..., 0:1, :]
        )
        self.assertAllClose(
            next_step[field][..., 1:5, :], log_t[field][..., 1:5, :]
        )

  @parameterized.named_parameters(
      ('AllowNewObjects', True), ('DontAllowNewObjects', False)
  )
  def test_update_state_with_dynamics_trajectory(self, allow_object_injection):
    sim_state = next(
        dataloader.simulator_state_generator(
            config=_config.DatasetConfig(
                path=TEST_DATA_PATH,
                max_num_objects=32,
                data_format=_config.DataFormat.TFRECORD,
                batch_dims=(),
            )
        )
    )

    current_traj = sim_state.current_sim_trajectory
    trajectory_update = datatypes.TrajectoryUpdate(
        x=jnp.ones_like(current_traj.x) * 1.0,
        y=jnp.ones_like(current_traj.y) * 2.0,
        vel_x=jnp.ones_like(current_traj.vel_x) * 4.0,
        vel_y=jnp.ones_like(current_traj.vel_y) * 5.0,
        yaw=jnp.ones_like(current_traj.yaw) * 6.0,
        valid=jnp.ones_like(current_traj.valid),
    )
    trajectory_update.validate()
    is_controlled = sim_state.object_metadata.is_sdc
    test_dynamics = TestDynamics(trajectory_update)
    updated_sim_traj = test_dynamics.forward(  # pytype: disable=wrong-arg-types  # jnp-type
        jnp.zeros_like(is_controlled),
        trajectory=sim_state.sim_trajectory,
        reference_trajectory=sim_state.log_trajectory,
        is_controlled=is_controlled,
        timestep=int(sim_state.timestep),
        allow_object_injection=allow_object_injection,
    )

    # These fields should be updated by the next trajectory.
    updated_sim_traj = datatypes.dynamic_slice(
        updated_sim_traj,
        start_index=int(sim_state.timestep) + 1,
        slice_size=1,
        axis=-1,
    )
    self.assertAllClose(
        updated_sim_traj.x[is_controlled], trajectory_update.x[is_controlled]
    )
    self.assertAllClose(
        updated_sim_traj.y[is_controlled], trajectory_update.y[is_controlled]
    )
    self.assertAllClose(
        updated_sim_traj.vel_x[is_controlled],
        trajectory_update.vel_x[is_controlled],
    )
    self.assertAllClose(
        updated_sim_traj.vel_y[is_controlled],
        trajectory_update.vel_y[is_controlled],
    )
    self.assertAllClose(
        updated_sim_traj.yaw[is_controlled],
        trajectory_update.yaw[is_controlled],
    )
    # These fields should be from the old current trajectory.
    self.assertAllClose(updated_sim_traj.z, current_traj.z)
    self.assertAllClose(updated_sim_traj.length, current_traj.length)
    self.assertAllClose(updated_sim_traj.width, current_traj.width)
    self.assertAllClose(updated_sim_traj.height, current_traj.height)
    # This field is updated from the next log trajectory.
    next_log_trajectory = datatypes.dynamic_slice(
        inputs=sim_state.log_trajectory,
        start_index=int(sim_state.timestep) + 1,
        slice_size=1,
        axis=-1,
    )
    self.assertAllEqual(
        updated_sim_traj.timestamp_micros, next_log_trajectory.timestamp_micros
    )
    # For non-controlled agents the above fields will be the next trajectory.
    self.assertAllClose(
        updated_sim_traj.x[~is_controlled],
        next_log_trajectory.x[~is_controlled],
    )
    self.assertAllClose(
        updated_sim_traj.y[~is_controlled],
        next_log_trajectory.y[~is_controlled],
    )
    self.assertAllClose(
        updated_sim_traj.vel_x[~is_controlled],
        next_log_trajectory.vel_x[~is_controlled],
    )
    self.assertAllClose(
        updated_sim_traj.vel_y[~is_controlled],
        next_log_trajectory.vel_y[~is_controlled],
    )
    self.assertAllClose(
        updated_sim_traj.yaw[~is_controlled],
        next_log_trajectory.yaw[~is_controlled],
    )
    # This field depends on the parameterization.
    if allow_object_injection:
      expected_valid = next_log_trajectory.valid
    else:
      current_valid = datatypes.dynamic_slice(
          sim_state.sim_trajectory,
          start_index=int(sim_state.timestep),
          slice_size=1,
          axis=-1,
      ).valid
      expected_valid = jnp.logical_and(current_valid, next_log_trajectory.valid)
    self.assertAllEqual(updated_sim_traj.valid, expected_valid)

  @parameterized.named_parameters(
      ('AllowNewObjects', True),
      ('DontAllowNewObjects', False),
  )
  def test_update_state_with_dynamics_trajectory_handles_valid(
      self, allow_object_injection
  ):
    data_config = _config.DatasetConfig(path=TEST_DATA_PATH, max_num_objects=5)
    sim_state = test_utils.make_zeros_state(data_config)
    sim_state = datatypes.update_state_by_log(sim_state, num_steps=10)

    current_valids = jnp.array([True, False, True, True, False])
    next_valids = jnp.array([True, True, False, False, False])
    is_controlled = jnp.array([True, True, True, False, True])
    action_valid = jnp.array([True, False, True, False, False])

    sim_current_valids = sim_state.sim_trajectory.valid.at[
        ..., sim_state.timestep
    ].set(current_valids)
    log_next_valids = sim_state.log_trajectory.valid.at[
        ..., sim_state.timestep + 1
    ].set(next_valids)
    sim_state = sim_state.replace(
        sim_trajectory=sim_state.sim_trajectory.replace(
            valid=sim_current_valids
        ),
        log_trajectory=sim_state.log_trajectory.replace(valid=log_next_valids),
    )
    current_traj = sim_state.current_sim_trajectory

    trajectory_update = datatypes.TrajectoryUpdate(
        x=jnp.ones_like(current_traj.x),
        y=jnp.ones_like(current_traj.y),
        vel_x=jnp.ones_like(current_traj.vel_x),
        vel_y=jnp.ones_like(current_traj.vel_y),
        yaw=jnp.ones_like(current_traj.yaw),
        valid=action_valid[..., jnp.newaxis],
    )
    test_dynamics = TestDynamics(trajectory_update)
    updated_sim_traj = test_dynamics.forward(  # pytype: disable=wrong-arg-types  # jnp-type
        jnp.zeros_like(is_controlled),
        trajectory=sim_state.sim_trajectory,
        reference_trajectory=sim_state.log_trajectory,
        is_controlled=is_controlled,
        timestep=int(sim_state.timestep),
        allow_object_injection=allow_object_injection,
    )

    if allow_object_injection:
      # Allows birth of the second object;
      expected_valid = current_valids.at[1].set(True)
      # Third object is controlled with valid current traj, thus will remain
      # valid; fourth object is non-controlled, and can die, resulting in False.
      expected_valid = expected_valid.at[3].set(False)
    else:
      # Non-controlled object can disappear, i.e. the forth object.
      expected_valid = current_valids.at[3].set(False)

    self.assertAllEqual(
        updated_sim_traj.valid[:, sim_state.timestep + 1], expected_valid
    )

  @parameterized.named_parameters(
      ('UseFallback', True),
      ('DontUseFallback', False),
  )
  def test_apply_trajectory_update_with_fallback(self, use_fallback):
    data_config = _config.DatasetConfig(path=TEST_DATA_PATH, max_num_objects=5)
    sim_state = test_utils.make_zeros_state(data_config)
    sim_state = datatypes.update_state_by_log(sim_state, num_steps=10)

    current_valids = jnp.array([True, False, True, True, False])
    next_valids = jnp.array([True, True, False, False, False])
    is_controlled = jnp.array([True, True, True, False, True])
    action_valid = jnp.array([True, False, True, False, False])

    sim_current_valids = sim_state.sim_trajectory.valid.at[
        ..., sim_state.timestep
    ].set(current_valids)
    log_next_valids = sim_state.log_trajectory.valid.at[
        ..., sim_state.timestep + 1
    ].set(next_valids)
    sim_state = sim_state.replace(
        sim_trajectory=sim_state.sim_trajectory.replace(
            valid=sim_current_valids
        ),
        log_trajectory=sim_state.log_trajectory.replace(valid=log_next_valids),
    )
    current_traj = sim_state.current_sim_trajectory

    trajectory_update = datatypes.TrajectoryUpdate(
        x=jnp.ones_like(current_traj.x),
        y=jnp.ones_like(current_traj.y),
        vel_x=jnp.ones_like(current_traj.vel_x),
        vel_y=jnp.ones_like(current_traj.vel_y),
        yaw=jnp.ones_like(current_traj.yaw),
        valid=action_valid[..., jnp.newaxis],
    )
    updated_sim_traj = abstract_dynamics.apply_trajectory_update_to_state(
        trajectory_update,
        sim_state.sim_trajectory,
        sim_state.log_trajectory,
        is_controlled=is_controlled,
        timestep=int(sim_state.timestep),
        use_fallback=use_fallback,
        allow_object_injection=False,
    )

    base_valid = (is_controlled & current_valids) | (
        ~is_controlled & next_valids
    )
    if use_fallback:
      # With fallback, agents are not invalidated when an action is invalid.
      expected_valid = base_valid
    else:
      # Without fallback, agents are invalidated when the action is invalid.
      expected_valid = base_valid & action_valid

    self.assertAllEqual(
        updated_sim_traj.valid[:, sim_state.timestep + 1], expected_valid
    )


if __name__ == '__main__':
  tf.test.main()
