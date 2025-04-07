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

import functools

import jax
from jax import numpy as jnp
import tensorflow as tf

from absl.testing import parameterized
from waymax import config as _config
from waymax import dataloader
from waymax import datatypes
from waymax import dynamics
from waymax import env as _env
from waymax.agents import expert
from waymax.utils import test_utils


TEST_DATA_PATH = test_utils.ROUTE_DATA_PATH
ROUTE_NUM_PATHS = test_utils.ROUTE_NUM_PATHS
ROUTE_NUM_POINTS_PER_PATH = test_utils.ROUTE_NUM_POINTS_PER_PATH


class UtilsTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.batch_dims = (2, 1)
    self.max_num_objects = 4
    self.dataset_config = _config.DatasetConfig(
        path=TEST_DATA_PATH,
        data_format=_config.DataFormat.TFRECORD,
        max_num_objects=4,
        batch_dims=(2, 1),
        include_sdc_paths=True,
        num_paths=ROUTE_NUM_PATHS,
        num_points_per_path=ROUTE_NUM_POINTS_PER_PATH,
    )
    self.state_t0 = next(
        dataloader.simulator_state_generator(config=self.dataset_config)
    )

  def test_rollout_log_by_expert_sdc_runs_end2end_with_real_data(self):
    rollout_num_steps = 2
    env_config = _config.EnvironmentConfig(
        init_steps=2, max_num_objects=self.dataset_config.max_num_objects
    )
    env = _env.MultiAgentEnvironment(
        dynamics_model=dynamics.DeltaGlobal(), config=env_config
    )
    rollout_fn = functools.partial(
        _env.rollout_log_by_expert_sdc,
        rollout_num_steps=rollout_num_steps,
        env=env,
        dynamics_model=env.dynamics,
    )
    # (obs, action, reward (Array), obs).
    out = rollout_fn(self.state_t0)
    # (rollout_num_steps, batch_dims).
    exp_shape = tuple([rollout_num_steps + 1]) + self.batch_dims
    self.assertEqual(out.observation.shape, exp_shape)
    self.assertEqual(
        out.observation.sim_trajectory.x.shape,
        exp_shape + (self.max_num_objects, 91),
    )
    self.assertEqual(
        out.action.shape, exp_shape + tuple([env_config.max_num_objects])
    )

  def test_rollout_result_matches_dynamics_and_reward(self):
    env = _env.MultiAgentEnvironment(
        dynamics_model=dynamics.DeltaGlobal(),
        config=_config.EnvironmentConfig(
            init_steps=2,
            max_num_objects=self.dataset_config.max_num_objects,
        ),
    )

    def _expert_action_fn(state, obs, rng):
      del obs, rng
      prev_sim_traj = datatypes.dynamic_slice(
          state.sim_trajectory, state.timestep, 1, axis=-1
      )
      logged_next_traj = datatypes.dynamic_slice(
          state.log_trajectory, state.timestep + 1, 1, axis=-1
      )
      combined_traj = jax.tree.map(
          lambda x, y: jnp.concatenate([x, y], axis=-1),
          prev_sim_traj,
          logged_next_traj,
      )
      return env.dynamics.inverse(
          combined_traj, metadata=state.object_metadata, timestep=0
      )

    result = _env.rollout(
        self.state_t0,
        expert.create_expert_actor(env.dynamics),
        env,
        rng=jax.random.PRNGKey(0),
        rollout_num_steps=2,
    )

    # Do a manual rollout as a reference.
    def _run_rollout(init_state):
      init_state = env.reset(init_state)
      state = init_state
      manual_rollout = []
      for _ in range(2):
        action = _expert_action_fn(state, None, None)
        next_state = env.step(state, action)
        output = _env.RolloutOutput(
            state=state,
            observation=env.observe(state),
            action=action,
            metrics={},
            reward=env.reward(state, action),
        )
        manual_rollout.append(output)
        state = next_state
      manual_rollout = jax.tree_util.tree_map(
          lambda *inputs: jnp.stack(inputs, axis=0), *manual_rollout
      )
      # Add back in the initial timestep.
      last_state = jax.tree_util.tree_map(
          lambda x: x[None], jax.tree_util.tree_map(jnp.asarray, next_state)
      )
      all_states = jax.tree_util.tree_map(
          lambda x, y: jnp.concatenate((x, y)), manual_rollout.state, last_state
      )
      last_observation = jax.tree_util.tree_map(
          lambda x: x[None], env.observe(next_state)
      )
      all_observations = jax.tree_util.tree_map(
          lambda x, y: jnp.concatenate((x, y)),
          manual_rollout.observation,
          last_observation,
      )
      invalid_action = datatypes.Action(
          data=jnp.zeros_like(action.data), valid=jnp.zeros_like(action.valid)
      )
      action = datatypes.Action(
          data=jnp.concatenate(
              (manual_rollout.action.data, invalid_action.data[None])
          ),
          valid=jnp.concatenate(
              (manual_rollout.action.valid, invalid_action.valid[None])
          ),
      )
      last_reard = env.reward(next_state, invalid_action)
      all_rewards = jnp.concatenate((manual_rollout.reward, last_reard[None]))
      return _env.RolloutOutput(
          state=all_states,
          observation=all_observations,
          action=action,
          metrics={},
          reward=all_rewards,
      )

    manual_rollout = _run_rollout(self.state_t0)

    def _assert_all_close(x, y):
      if x is None:
        self.assertIsNone(y)
      else:
        self.assertAllClose(x, y)

    with self.subTest('manual rollout produces same results as scan'):
      jax.tree_util.tree_map(
          _assert_all_close, result.state, manual_rollout.state
      )
      self.assertAllClose(result.action, manual_rollout.action)
      self.assertAllClose(result.reward, manual_rollout.reward)

  def test_sdc_environment_rollout_without_observation_function_correct_shape(
      self,
  ):
    dataset_config = _config.DatasetConfig(
        path=test_utils.ROUTE_DATA_PATH,
        data_format=_config.DataFormat.TFRECORD,
    )
    state_t0 = next(dataloader.simulator_state_generator(config=dataset_config))

    env = _env.PlanningAgentEnvironment(
        dynamics_model=dynamics.DeltaGlobal(),
        config=_config.EnvironmentConfig(),
    )
    result = _env.rollout(
        state_t0,
        expert.create_expert_actor(env.dynamics),
        env,
        rng=jax.random.PRNGKey(0),
        rollout_num_steps=2,
    )
    self.assertEqual(
        result.observation.shape,
        tuple([3]),
    )


if __name__ == '__main__':
  tf.test.main()
