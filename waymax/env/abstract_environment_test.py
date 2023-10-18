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
import tensorflow as tf
from waymax import config as _config
from waymax import datatypes
from waymax.env import abstract_environment
from waymax.env import typedefs as types
from waymax.utils import test_utils


class MockAbstractEnvironment(abstract_environment.AbstractEnvironment):
  """Mocked version of the abstract environment that tests default functions."""

  def reset(self, state: datatypes.SimulatorState) -> datatypes.SimulatorState:
    """Initializes a simulation state."""
    raise NotImplementedError()

  def step(
      self, state: datatypes.SimulatorState, action: datatypes.Action
  ) -> datatypes.SimulatorState:
    """Advances the simulation by one timestep."""
    raise NotImplementedError()

  def action_spec(self) -> datatypes.Action:
    """Returns the action specs of the environment without batch dimension.

    Returns:
      The action specs represented as a nested datatypes.structure where the
      leaves are instances of specs.Array.
    """
    raise NotImplementedError()

  def reward_spec(self) -> specs.Array:
    """Returns the reward specs of the environment without batch dimension."""
    raise NotImplementedError()

  def discount_spec(self) -> specs.BoundedArray:
    """Returns the discount specs of the environment without batch dimension."""
    raise NotImplementedError()

  def observation_spec(self) -> types.PyTree:
    """Returns the observation specs."""
    raise NotImplementedError()

  def reward(
      self, state: datatypes.SimulatorState, action: datatypes.Action
  ) -> jax.Array:
    """Computes the reward for a transition."""
    raise NotImplementedError()

  def metrics(self, state: datatypes.SimulatorState) -> types.Metrics:
    """Not implemented metrics function."""
    raise NotImplementedError()

  def observe(self, state: datatypes.SimulatorState) -> types.Observation:
    """Not implemented observe function."""
    raise NotImplementedError()


class AbstractEnvironmentTest(tf.test.TestCase):

  def test_truncation_produces_correct_result(self):
    simulator_mock = MockAbstractEnvironment()
    dataset_config = _config.DatasetConfig(path='')
    zeros_simulator_state = test_utils.make_zeros_state(dataset_config)
    # Update this so there's a few simulator state's.
    zeros_simulator_state = zeros_simulator_state.replace(
        timestep=zeros_simulator_state.remaining_timesteps - 1
    )
    self.assertAllEqual(
        simulator_mock.truncation(zeros_simulator_state),
        tf.zeros_like(simulator_mock.truncation(zeros_simulator_state)),
    )

    # Now it should be the end of the simulation.
    zeros_simulator_state = zeros_simulator_state.replace(
        timestep=zeros_simulator_state.timestep + 1
    )
    self.assertAllEqual(
        simulator_mock.truncation(zeros_simulator_state),
        tf.ones_like(simulator_mock.truncation(zeros_simulator_state)),
    )


if __name__ == '__main__':
  tf.test.main()
