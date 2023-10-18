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

"""Tests for the abstract reward function."""
import jax
import jax.numpy as jnp
import tensorflow as tf

from waymax import dataloader
from waymax import datatypes
from waymax.rewards import abstract_reward_function
from waymax.utils import test_utils

TEST_DATA_PATH = test_utils.ROUTE_DATA_PATH


class SimpleReward(abstract_reward_function.AbstractRewardFunction):

  def compute(
      self,
      simulator_state: datatypes.SimulatorState,
      action: datatypes.Action,
      agent_mask: jax.Array,
  ) -> jax.Array:
    return 1.23


class NoDefinedComputeReward(abstract_reward_function.AbstractRewardFunction):

  def __init__(self):
    pass


class AbstractRewardFunctionTest(tf.test.TestCase):

  def test_abstract_reward_subclass_instantiates(self):
    dataset = test_utils.make_test_dataset()
    data_dict = next(dataset.as_numpy_iterator())
    state = dataloader.simulator_state_from_womd_dict(data_dict, time_key='all')
    agent_mask = jnp.ones(128)
    action = datatypes.Action(data=jnp.array(()), valid=jnp.array(()))

    simple_reward = SimpleReward()
    reward = simple_reward.compute(state, action, agent_mask)

    self.assertEqual(reward, 1.23)

  def test_reward_function_no_defined_compute_raises(self):
    self.assertRaises(TypeError, NoDefinedComputeReward)


if __name__ == '__main__':
  tf.test.main()
