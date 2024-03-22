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

"""Tests linear combination reward function."""

import jax.numpy as jnp
import numpy as np
import tensorflow as tf

from waymax import config as _config
from waymax import datatypes
from waymax.rewards import linear_combination_reward
from waymax.utils import test_utils


class LinearCombinationRewardTest(tf.test.TestCase):

  def test_linear_combination_results_correct(self):
    combination_config = _config.LinearCombinationRewardConfig(
        {'overlap': -1.0, 'offroad': 1.0}
    )
    combination_reward_function = (
        linear_combination_reward.LinearCombinationReward(combination_config)
    )

    overlap_config = _config.LinearCombinationRewardConfig({'overlap': -1.0})
    negative_overlap_reward_function = (
        linear_combination_reward.LinearCombinationReward(overlap_config)
    )

    offroad_config = _config.LinearCombinationRewardConfig({'offroad': 1.0})
    offroad_reward_function = linear_combination_reward.LinearCombinationReward(
        offroad_config
    )

    state = test_utils.simulator_state_with_overlap()
    agent_mask = jnp.ones(
        state.sim_trajectory.shape[0],
    )
    action = datatypes.Action(data=jnp.array(()), valid=jnp.array(()))

    combination_reward = combination_reward_function.compute(
        state, action, agent_mask
    )
    negative_overlap_reward = negative_overlap_reward_function.compute(
        state, action, agent_mask
    )
    offroad_reward = offroad_reward_function.compute(state, action, agent_mask)
    manual_combination_reward = negative_overlap_reward + offroad_reward

    with self.subTest('combination_equals_manual_sum'):
      self.assertAllClose(combination_reward, manual_combination_reward)

    with self.subTest('combination_equals_manual_entry'):
      self.assertAllClose(combination_reward, [0, 0, -1, 0, 0, -0, -0])

    with self.subTest('overlap_reward_is_correct'):
      self.assertAllClose(negative_overlap_reward, [-1] * 7)

    with self.subTest('offroad_reward_is_correct'):
      self.assertAllClose(offroad_reward, [1, 1, 0, 1, 1, 1, 1])

  def test_returns_zero_for_invalid_trajectory(self):
    combination_config = _config.LinearCombinationRewardConfig({
        'overlap': -1.0,
    })
    combination_reward_function = (
        linear_combination_reward.LinearCombinationReward(combination_config)
    )

    state = test_utils.simulator_state_with_overlap()
    invalid_sim_trajectory = state.sim_trajectory
    invalid_sim_trajectory = invalid_sim_trajectory.replace(
        valid=jnp.zeros_like(invalid_sim_trajectory.valid)
    )
    state = state.replace(sim_trajectory=invalid_sim_trajectory)
    agent_mask = jnp.ones(
        state.sim_trajectory.shape[0],
    )
    action = datatypes.Action(data=jnp.array(()), valid=jnp.array(()))

    combination_reward = combination_reward_function.compute(
        state, action, agent_mask
    )
    expected_combination_reward = jnp.zeros_like(agent_mask)
    np.testing.assert_array_almost_equal(
        combination_reward, expected_combination_reward
    )

  def test_invalid_metric_raises(self):
    config = _config.LinearCombinationRewardConfig({'fake_name_metric': 1.0})
    self.assertRaises(
        ValueError, linear_combination_reward.LinearCombinationReward, config
    )


if __name__ == '__main__':
  tf.test.main()
