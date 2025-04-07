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

import dm_env
import jax.numpy as jnp
import tensorflow as tf

from absl.testing import parameterized
from waymax import config as _config
from waymax import dataloader
from waymax import dynamics
from waymax.env import base_environment
from waymax.env import errors
from waymax.env import planning_agent_environment
from waymax.env.wrappers import dm_env_wrapper
from waymax.utils import test_utils

_TEST_DATA_PATH = test_utils.ROUTE_DATA_PATH


class WaymaxDMEnvTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.dataset_config = _config.DatasetConfig(
        path=_TEST_DATA_PATH, data_format=_config.DataFormat.TFRECORD
    )
    dataset_iter = dataloader.simulator_state_generator(self.dataset_config)

    # TODO(b/251258357) Update environment tests to use other dynamics.
    # Uses large init step (in this case only 2 steps left) to reduce test time.
    self.env_config = _config.EnvironmentConfig(init_steps=88)
    self.multi_stateless_env = base_environment.MultiAgentEnvironment(
        dynamics_model=dynamics.DeltaGlobal(), config=self.env_config
    )
    self.multi_env = dm_env_wrapper.DMEnvWrapper(
        dataset_iter, self.multi_stateless_env
    )
    self.single_stateless_env = (
        planning_agent_environment.PlanningAgentEnvironment(
            dynamics_model=dynamics.DeltaGlobal(), config=self.env_config
        )
    )
    self.single_env = dm_env_wrapper.DMEnvWrapper(
        dataset_iter, self.single_stateless_env
    )

  @parameterized.parameters(True, False)
  def test_observation_matches_spec(self, multiagent: bool):
    env = self.multi_env if multiagent else self.single_env
    timestep = env.reset()
    obs_spec = env.observation_spec()
    observation = timestep.observation
    self.assertEqual(
        self.dataset_config.batch_dims + obs_spec.shape, observation.shape
    )

  @parameterized.parameters(True, False)
  def test_reset_returns_first_timestep(self, multiagent: bool):
    env = self.multi_env if multiagent else self.single_env
    timestep = env.reset()
    self.assertEqual(timestep.step_type, dm_env.StepType.FIRST)
    self.assertEqual(
        env.simulation_state.timestep + 1, self.env_config.init_steps
    )

  @parameterized.parameters(True, False)
  def test_step_returns_mid_timestep(self, multiagent: bool):
    env = self.multi_env if multiagent else self.single_env
    env.reset()
    action = env.action_spec().minimum
    timestep = env.step(action)
    self.assertEqual(timestep.step_type, dm_env.StepType.MID)
    self.assertGreater(env.simulation_state.remaining_timesteps, 0)

  @parameterized.parameters(True, False)
  def test_end_of_episode_returns_last_timestep(self, multiagent: bool):
    multiagent = True
    env = self.multi_env if multiagent else self.single_env
    env.reset()
    expected_length = env.simulation_state.remaining_timesteps
    action = env.action_spec().minimum
    for _ in range(expected_length):
      timestep = env.step(action)
    with self.subTest('dm timestep arrives last'):
      self.assertEqual(timestep.step_type, dm_env.StepType.LAST)
    with self.subTest('simulation state reaches the end'):
      self.assertEqual(env.simulation_state.remaining_timesteps, 0)
    with self.assertRaises(errors.EpisodeAlreadyFinishedError):
      env.step(action)

  @parameterized.parameters(True, False)
  def test_raises_error_when_stepping_before_initialization(
      self, multiagent: bool
  ):
    env = self.multi_env if multiagent else self.single_env
    action = env.action_spec().minimum
    with self.assertRaises(errors.SimulationNotInitializedError):
      env.step(action)

  @parameterized.parameters(True, False)
  def test_raises_error_when_accessing_simulation_state_before_init(
      self, multiagent: bool
  ):
    env = self.multi_env if multiagent else self.single_env
    with self.assertRaises(errors.SimulationNotInitializedError):
      env.simulation_state()

  @parameterized.parameters(((),), ((2, 1),))
  def test_env_is_compatible_with_batch_dims(self, batch_dims):
    config = _config.DatasetConfig(
        path=_TEST_DATA_PATH,
        batch_dims=batch_dims,
        data_format=_config.DataFormat.TFRECORD,
    )
    dataset_iter = dataloader.simulator_state_generator(config)
    env = dm_env_wrapper.DMEnvWrapper(
        dataset_iter,
        self.single_stateless_env,
        self.single_stateless_env.observe,
    )
    action = env.action_spec().minimum
    # Adding batch dimensions if needed.
    for i in reversed(batch_dims):
      action = jnp.repeat(action[jnp.newaxis], i, axis=0)
    env.reset()
    env.step(action)

  def test_sdc_dm_environment_has_correct_batch_dims(self):
    batch_dims = (2, 1)
    dynamics_model = dynamics.DeltaGlobal()
    max_num_objects = 32
    dataset_config = _config.DatasetConfig(
        path=_TEST_DATA_PATH,
        data_format=_config.DataFormat.TFRECORD,
        max_num_objects=max_num_objects,
        batch_dims=batch_dims,
    )
    env_config = _config.EnvironmentConfig(
        max_num_objects=max_num_objects,
        controlled_object=_config.ObjectType.SDC,
    )

    env = dm_env_wrapper.make_sdc_dm_environment(
        dynamics_model=dynamics_model,
        data_config=dataset_config,
        env_config=env_config,
    )
    env.reset()
    action = env.action_spec().minimum
    for i in reversed(batch_dims):
      action = jnp.repeat(action[jnp.newaxis], i, axis=0)
    ts = env.step(action)

    self.assertEqual(ts.observation.shape, batch_dims)
    self.assertEqual(ts.reward.shape, batch_dims)

  def test_sdc_dm_environment_has_correct_shape_with_no_observe(self):
    batch_dims = (2, 1)
    dynamics_model = dynamics.DeltaGlobal()
    max_num_objects = 32
    dataset_config = _config.DatasetConfig(
        path=_TEST_DATA_PATH,
        data_format=_config.DataFormat.TFRECORD,
        max_num_objects=max_num_objects,
        batch_dims=batch_dims,
    )
    env_config = _config.EnvironmentConfig(
        max_num_objects=max_num_objects,
        controlled_object=_config.ObjectType.SDC,
    )

    env = dm_env_wrapper.make_sdc_dm_environment(
        dynamics_model=dynamics_model,
        data_config=dataset_config,
        env_config=env_config,
    )
    env.reset()
    action = env.action_spec().minimum
    for i in reversed(batch_dims):
      action = jnp.repeat(action[jnp.newaxis], i, axis=0)
    ts = env.step(action)

    self.assertEqual(ts.observation.shape, batch_dims)
    self.assertEqual(ts.reward.shape, batch_dims)

  def test_observation_matches_spec_with_no_observe(self):
    batch_dims = (2, 1)
    dynamics_model = dynamics.DeltaGlobal()
    max_num_objects = 32
    dataset_config = _config.DatasetConfig(
        path=_TEST_DATA_PATH,
        data_format=_config.DataFormat.TFRECORD,
        max_num_objects=max_num_objects,
        batch_dims=batch_dims,
    )
    env_config = _config.EnvironmentConfig(
        max_num_objects=max_num_objects,
        controlled_object=_config.ObjectType.SDC,
    )

    env = dm_env_wrapper.make_sdc_dm_environment(
        dynamics_model=dynamics_model,
        data_config=dataset_config,
        env_config=env_config,
    )
    obs_spec = env.observation_spec()
    observation = env.reset().observation
    self.assertEqual(batch_dims + obs_spec.shape, observation.shape)


if __name__ == '__main__':
  tf.test.main()
