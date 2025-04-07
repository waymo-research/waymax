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

from typing import Optional

import chex
import jax
from jax import numpy as jnp
import tensorflow as tf
from waymax import config as _config
from waymax import dataloader
from waymax import datatypes
from waymax import dynamics
from waymax.agents import actor_core
from waymax.agents import sim_agent
from waymax.env import planning_agent_environment
from waymax.utils import test_utils

from absl.testing import parameterized

TEST_DATA_PATH = test_utils.ROUTE_DATA_PATH
ROUTE_NUM_PATHS = test_utils.ROUTE_NUM_PATHS
ROUTE_NUM_POINTS_PER_PATH = test_utils.ROUTE_NUM_POINTS_PER_PATH


class PlanningAgentEnvironmentTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    super().setUp()

    self.env_config = _config.EnvironmentConfig(init_steps=10)
    # TODO(b/251258357) Update environment tests to use other dynamics.
    self.env = planning_agent_environment.PlanningAgentEnvironment(
        dynamics_model=dynamics.DeltaGlobal(), config=self.env_config
    )
    action_spec = self.env.action_spec()
    self.sample_action = datatypes.Action(
        data=jnp.zeros(action_spec.data.shape, dtype=jnp.float32),
        valid=jnp.ones(action_spec.valid.shape, dtype=jnp.bool_),
    )
    self.dataset_config = _config.DatasetConfig(
        path=TEST_DATA_PATH,
        data_format=_config.DataFormat.TFRECORD,
        include_sdc_paths=True,
        num_paths=ROUTE_NUM_PATHS,
        num_points_per_path=ROUTE_NUM_POINTS_PER_PATH,
    )
    dataset_iter = dataloader.simulator_state_generator(self.dataset_config)
    self.sample_state = next(dataset_iter)

  def test_initial_state_sets_initial_timestep(self):
    init_state = self.env.reset(self.sample_state)
    self.assertEqual(init_state.timestep, 9)

  def test_transition_advances_timestep(self):
    init_state = self.env.reset(self.sample_state)
    next_state = self.env.step(init_state, self.sample_action)
    self.assertEqual(next_state.timestep, init_state.timestep + 1)

  def test_inverse_has_correct_shape(self):
    inverse_action = self.env.dynamics.inverse(
        self.sample_state.log_trajectory, self.sample_state.object_metadata, 0
    )
    self.assertEqual(
        inverse_action.data.shape, self.env.action_spec().data.shape
    )

  @parameterized.named_parameters(
      ('compute_reward', True), ('dont_compute_reward', False)
  )
  def test_reward_has_correct_shape(self, compute_reward: bool):
    env_config = _config.EnvironmentConfig(
        init_steps=10, compute_reward=compute_reward
    )
    env = planning_agent_environment.PlanningAgentEnvironment(
        dynamics_model=dynamics.DeltaGlobal(), config=env_config
    )
    sample_state = env.reset(self.sample_state)
    reward = env.reward(sample_state, self.sample_action)
    self.assertAllEqual(reward.shape, ())
    self.assertAllEqual(reward.dtype, jnp.float32)

  def test_metric_has_correct_shape(self):
    env_config = _config.EnvironmentConfig(
        init_steps=10,
        metrics=_config.MetricsConfig(
            metrics_to_run=(
                'log_divergence',
                'overlap',
                'offroad',
                'sdc_wrongway',
                'sdc_progression',
                'sdc_off_route',
                'kinematic_infeasibility',
            ),
        ),
    )
    env = planning_agent_environment.PlanningAgentEnvironment(
        dynamics_model=dynamics.DeltaGlobal(), config=env_config
    )
    sample_state = env.reset(self.sample_state)
    metrics_dict = env.metrics(sample_state)
    num_metrics = 7
    metric_shape_targets = tuple([() for _ in range(num_metrics)])
    metric_shapes = [v.shape for _, v in metrics_dict.items()]
    with self.subTest('all_metrics_are_predicted'):
      self.assertLen(metrics_dict.keys(), num_metrics)
    with self.subTest('all_metrics_are_populated'):
      self.assertAllEqual(metric_shapes, metric_shape_targets)

  @parameterized.named_parameters(('without_keys', None), ('with_keys', 100))
  def test_planning_agent_environment_with_sim_agents_works(self, key):
    state = test_utils.make_zeros_state(self.dataset_config)
    state = planning_agent_environment.PlanningAgentSimulatorState(**state)
    state = state.replace(timestep=0)
    env = planning_agent_environment.PlanningAgentEnvironment(
        dynamics_model=dynamics.DeltaGlobal(),
        config=self.env_config,
        sim_agent_actors=[constant_velocity_actor()],
        sim_agent_params=[{}],
    )
    key = jax.random.PRNGKey(key) if key is not None else key
    state = env.reset(state, rng=key)
    action_spec = self.env.action_spec()
    action = datatypes.Action(
        data=jnp.array([0.7, 0.8, 0.05]),
        valid=jnp.ones(action_spec.valid.shape, dtype=jnp.bool_),
    )
    state = env.step(state, action, rng=key)

    traj = state.current_sim_trajectory
    is_sdc = state.object_metadata.is_sdc
    with self.subTest('PlanningAgentStateIsCorrect'):
      self.assertAllClose(traj.x[is_sdc], jnp.ones_like(traj.x[is_sdc]) * 0.7)
      self.assertAllClose(traj.y[is_sdc], jnp.ones_like(traj.x[is_sdc]) * 0.8)
      self.assertAllClose(
          traj.yaw[is_sdc], jnp.ones_like(traj.x[is_sdc]) * 0.05
      )
    with self.subTest('SimAgentStateIsCorrect'):
      self.assertAllClose(traj.x[~is_sdc], jnp.ones_like(traj.x[~is_sdc]) * 0.5)
      self.assertAllClose(traj.y[~is_sdc], jnp.ones_like(traj.x[~is_sdc]) * 0.6)
      self.assertAllClose(
          traj.yaw[~is_sdc], jnp.ones_like(traj.x[~is_sdc]) * 0.1
      )
      self.assertEqual(
          state.sim_agent_actor_states, [ConstantSimAgentState(state_num=1)]
      )

  def test_planning_agent_environment_raises_with_sim_actor_params(self):
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        'Number of sim agents must match number of sim agent params.',
    ):
      planning_agent_environment.PlanningAgentEnvironment(
          dynamics_model=dynamics.DeltaGlobal(),
          config=self.env_config,
          sim_agent_actors=[constant_velocity_actor()],
          sim_agent_params=[{}, {}],
      )

  def test_initialized_overlap_mask(self):
    log_traj = datatypes.Trajectory.zeros(shape=(3, 2))
    log_traj = log_traj.replace(
        x=jnp.array([[0, 1], [0, 2], [1, 3]]),
        y=jnp.array([[0, 1], [0, 2], [1, 3]]),
        z=jnp.array([[0, 1], [0, 2], [1, 3]]),
        length=jnp.array([[1, 1], [1, 1], [1, 1]]),
        width=jnp.array([[1, 1], [1, 1], [1, 1]]),
        height=jnp.array([[1, 1], [1, 1], [1, 1]]),
    )
    overlaps = planning_agent_environment._initialized_overlap(log_traj)
    # Agents 1 and 2 are overlapped on the first timestep.
    self.assertAllClose(overlaps, jnp.array([True, True, False]))


class ConstantSimAgentActor(sim_agent.SimAgentActor):
  """Sim agent actor that always returns the same output."""

  DELTA_X = 0.5
  DELTA_Y = 0.6
  DELTA_YAW = 0.1

  def update_trajectory(
      self, state: datatypes.SimulatorState
  ) -> datatypes.TrajectoryUpdate:
    """Just add the constant values to the pose."""
    traj = state.current_sim_trajectory
    return datatypes.TrajectoryUpdate(
        x=traj.x + self.DELTA_X,
        y=traj.y + self.DELTA_Y,
        yaw=traj.yaw + self.DELTA_YAW,
        vel_x=traj.vel_x,
        vel_y=traj.vel_y,
        valid=traj.valid,
    )


@chex.dataclass(frozen=True)
class ConstantSimAgentState:
  state_num: int = 0


def constant_velocity_actor() -> actor_core.WaymaxActorCore:
  agent = ConstantSimAgentActor()

  def init(rng, state: datatypes.SimulatorState) -> ConstantSimAgentState:
    del rng, state
    return ConstantSimAgentState()

  def select_action(
      params: Optional[actor_core.Params],
      state: datatypes.SimulatorState,
      actor_state: ConstantSimAgentState,
      rng: Optional[jax.Array] = None,
  ) -> actor_core.WaymaxActorOutput:
    del params, rng
    action = agent.update_trajectory(state).as_action()
    output = actor_core.WaymaxActorOutput(
        action=action,
        actor_state=actor_state.replace(state_num=actor_state.state_num + 1),
        is_controlled=~state.object_metadata.is_sdc,
    )
    output.validate()
    return output

  return actor_core.actor_core_factory(
      init=init,
      select_action=select_action,
      name='constant_vel',
  )


if __name__ == '__main__':
  tf.test.main()
