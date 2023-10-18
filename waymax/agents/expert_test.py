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

"""Tests for expert."""
import jax
import jax.numpy as jnp
import tensorflow as tf
import tree

from waymax import config as _config
from waymax import dataloader
from waymax import datatypes
from waymax import dynamics
from waymax.agents import expert
from waymax.utils import test_utils


class ExpertTest(tf.test.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls._data_config = _config.DatasetConfig(path='')
    cls._dynamics_model = dynamics.DeltaGlobal()
    cls._expert_actor_sdc = expert.create_expert_actor(cls._dynamics_model)

  def test_select_action_produces_correct_output_for_sdc(self):
    simulator_state = test_utils.make_zeros_state(self._data_config)
    log_trajectory = simulator_state.log_trajectory
    one_step_traj = tree.map_structure(
        lambda x: jnp.ones_like(x[..., 0:1]), log_trajectory
    )

    def update_simulator_state(
        vals: tuple[float, float, float], index
    ) -> datatypes.SimulatorState:
      new_x, new_y, new_yaw = vals
      traj = one_step_traj.replace(
          x=jnp.ones_like(one_step_traj.x) * new_x,
          y=jnp.ones_like(one_step_traj.y) * new_y,
          yaw=jnp.ones_like(one_step_traj.yaw) * new_yaw,
      )
      return simulator_state.replace(
          log_trajectory=datatypes.update_by_slice_in_dim(
              inputs=simulator_state.log_trajectory,
              updates=traj,
              inputs_start_idx=index,
              slice_size=1,
              axis=-1,
          ),
          sim_trajectory=datatypes.update_by_slice_in_dim(
              inputs=simulator_state.sim_trajectory,
              updates=traj,
              inputs_start_idx=index,
              slice_size=1,
              axis=-1,
          ),
      )

    simulator_state = update_simulator_state((0.0, 0.0, 0.0), 0)
    simulator_state = update_simulator_state((1.0, 1.0, 0.1), 1)
    simulator_state = update_simulator_state((3.0, 3.0, 0.11), 2)

    actor_init_state = self._expert_actor_sdc.init(
        jax.random.PRNGKey(0), simulator_state
    )
    first_output = self._expert_actor_sdc.select_action(
        None, simulator_state, actor_init_state, jax.random.PRNGKey(0)
    )
    simulator_state = simulator_state.replace(timestep=1)
    second_output = self._expert_actor_sdc.select_action(
        None, simulator_state, first_output.actor_state, jax.random.PRNGKey(0)
    )

    def get_action(
        shape_prefix: tuple[int, ...], actions: tuple[float, float, float]
    ) -> datatypes.Action:
      dx, dy, dtheta = actions
      action_data = jnp.stack(
          [
              jnp.ones(shape_prefix, dtype=jnp.float32) * dx,
              jnp.ones(shape_prefix, dtype=jnp.float32) * dy,
              jnp.ones(shape_prefix, dtype=jnp.float32) * dtheta,
          ],
          axis=-1,
      )
      return datatypes.Action(
          data=action_data, valid=jnp.ones(shape_prefix + (1,), dtype=jnp.bool_)
      )

    self.assertAllClose(
        get_action(first_output.action.shape, (1.0, 1.0, 0.1)),
        first_output.action,
    )
    self.assertAllClose(
        get_action(first_output.action.shape, (2.0, 2.0, 0.01)),
        second_output.action,
    )

  def test_select_action_produces_correct_output_for_controlled_objects(self):
    max_num_objects = 32
    config = _config.DatasetConfig(
        path=test_utils.ROUTE_DATA_PATH,
        data_format=_config.DataFormat.TFRECORD,
        max_num_objects=max_num_objects,
    )
    simulator_state = next(dataloader.simulator_state_generator(config=config))
    ids = jnp.arange(max_num_objects)
    is_controlled_func = lambda state: (ids >= 3) & (ids < 8)
    gt_xy_yaw = jnp.stack(
        [
            simulator_state.log_trajectory.x,
            simulator_state.log_trajectory.y,
            simulator_state.log_trajectory.yaw,
        ],
        axis=-1,
    )
    gt_valid = simulator_state.log_trajectory.valid
    gt_first_action = gt_xy_yaw[3:8, 1] - gt_xy_yaw[3:8, 0]
    gt_first_valid = gt_valid[3:8, :1] & gt_valid[3:8, 1:2]
    gt_second_action = gt_xy_yaw[3:8, 2] - gt_xy_yaw[3:8, 1]
    gt_second_valid = gt_valid[3:8, 1:2] & gt_valid[3:8, 2:3]

    expert_actor = expert.create_expert_actor(
        self._dynamics_model, is_controlled_func
    )
    actor_init_state = expert_actor.init(jax.random.PRNGKey(0), simulator_state)
    first_output = expert_actor.select_action(
        None, simulator_state, actor_init_state, jax.random.PRNGKey(0)
    )
    simulator_state = datatypes.update_state_by_log(
        simulator_state, num_steps=1
    )
    second_output = expert_actor.select_action(
        None, simulator_state, first_output.actor_state, jax.random.PRNGKey(0)
    )

    self.assertAllClose(first_output.action.data[3:8], gt_first_action)
    self.assertAllClose(first_output.action.valid[3:8], gt_first_valid)
    self.assertAllClose(second_output.action.data[3:8], gt_second_action)
    self.assertAllClose(second_output.action.valid[3:8], gt_second_valid)


if __name__ == '__main__':
  tf.test.main()
