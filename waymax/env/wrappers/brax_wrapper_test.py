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
from jax import numpy as jnp
import tensorflow as tf

from absl.testing import parameterized
from waymax import config as _config
from waymax import dataloader
from waymax import dynamics
from waymax.env import base_environment
from waymax.env import planning_agent_environment
from waymax.env.wrappers import brax_wrapper
from waymax.utils import test_utils

_TEST_DATA_PATH = test_utils.ROUTE_DATA_PATH


class BraxWrapperTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self.dataset_config = _config.DatasetConfig(
        path=_TEST_DATA_PATH,
        batch_dims=(),
        data_format=_config.DataFormat.TFRECORD,
    )
    self.dataset_iter = dataloader.simulator_state_generator(
        self.dataset_config
    )
    self.state_0 = next(self.dataset_iter)

    self.env_config = _config.EnvironmentConfig(init_steps=88)
    multi_stateless_env = base_environment.MultiAgentEnvironment(
        dynamics_model=dynamics.DeltaGlobal(), config=self.env_config
    )
    self.multi_env = brax_wrapper.BraxWrapper(
        multi_stateless_env,
        multi_stateless_env.dynamics,
        multi_stateless_env.config,
    )
    single_stateless_env = planning_agent_environment.PlanningAgentEnvironment(
        dynamics_model=dynamics.DeltaGlobal(), config=self.env_config
    )
    self.single_env = brax_wrapper.BraxWrapper(
        single_stateless_env,
        single_stateless_env.dynamics,
        single_stateless_env.config,
    )

  @parameterized.parameters(True, False)
  def test_reset_returns_first_timestep(self, multi=False):
    env = self.multi_env if multi else self.single_env
    reset_ts = env.reset(self.state_0)
    self.assertAllClose(reset_ts.discount, 1.0)
    self.assertAllEqual(reset_ts.done, False)

  @parameterized.parameters(True, False)
  def test_step_advances_timestep(self, multi=False):
    env = self.multi_env if multi else self.single_env
    reset_ts = env.reset(self.state_0)
    action = jax.tree.map(
        lambda x: jnp.zeros(x.shape, dtype=x.dtype), env.action_spec()
    )
    next_ts = env.step(reset_ts, action)
    self.assertEqual(next_ts.state.timestep, reset_ts.state.timestep + 1)
    self.assertEqual(next_ts.state.sim_trajectory.shape, (128, 91))

  @parameterized.parameters(((),), ((2, 1),))
  def test_env_is_compatible_with_batch_dims(self, batch_dims):
    config = _config.DatasetConfig(
        path=_TEST_DATA_PATH,
        batch_dims=batch_dims,
        data_format=_config.DataFormat.TFRECORD,
    )
    dataset_iter = dataloader.simulator_state_generator(config)
    action = jax.tree.map(
        lambda x: jnp.zeros(x.shape, dtype=x.dtype),
        self.multi_env.action_spec(),
    )
    # Adding batch dimensions if needed.
    for ndims in reversed(batch_dims):
      # pylint: disable=cell-var-from-loop
      action = jax.tree.map(
          lambda x: jnp.repeat(x[jnp.newaxis], ndims, axis=0), action
      )
    new_state = self.multi_env.reset(next(dataset_iter))
    self.multi_env.step(new_state, action)


if __name__ == '__main__':
  tf.test.main()
