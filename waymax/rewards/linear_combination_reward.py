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

"""Reward functions for the Waymax environment."""
import jax
import jax.numpy as jnp
from waymax import config as _config
from waymax import datatypes
from waymax import metrics
from waymax.rewards import abstract_reward_function


class LinearCombinationReward(abstract_reward_function.AbstractRewardFunction):
  """Reward function that performs a linear combination of metrics."""

  def __init__(self, config: _config.LinearCombinationRewardConfig):
    _validate_reward_metrics(config)

    self._config = config
    self._metrics_config = _linear_config_to_metric_config(self._config)

  def compute(
      self,
      simulator_state: datatypes.SimulatorState,
      action: datatypes.Action,
      agent_mask: jax.Array,
  ) -> jax.Array:
    """Computes the reward as a linear combination of metrics.

    Args:
      simulator_state: State of the Waymax environment.
      action: Action taken to control the agent(s) (..., num_objects,
        action_space).
      agent_mask: Binary mask indicating which agent inputs are valid (...,
        num_objects).

    Returns:
      An array of rewards, where there is one reward per agent
      (..., num_objects).
    """
    del action  # unused
    all_metrics = metrics.run_metrics(simulator_state, self._metrics_config)

    reward = jnp.zeros_like(agent_mask)
    for reward_metric_name, reward_weight in self._config.rewards.items():
      metric_all_agents = all_metrics[reward_metric_name].masked_value()
      metric = metric_all_agents * agent_mask
      reward += metric * reward_weight

    return reward


def _validate_reward_metrics(config: _config.LinearCombinationRewardConfig):
  """Checks that all metrics in the RewardConfigs are valid."""
  metric_names = metrics.get_metric_names()
  for reward_metric_name in config.rewards.keys():
    if reward_metric_name not in metric_names:
      raise ValueError(
          f'Invalid metric name {reward_metric_name} was given to '
          'the rewards config. Only the following metrics are '
          f'supported: {metric_names}'
      )


def _linear_config_to_metric_config(
    config: _config.LinearCombinationRewardConfig,
) -> _config.MetricsConfig:
  """Converts a LinearCombinationRewardConfig into a MetricsConfig."""
  return _config.MetricsConfig(metrics_to_run=tuple(config.rewards.keys()))
