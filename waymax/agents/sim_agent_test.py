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
import tensorflow as tf

from waymax import config as _config
from waymax import datatypes
from waymax.agents import sim_agent

from waymax.utils import test_utils


class SimAgentTest(tf.test.TestCase):

  def test_actor_core_controlled_objects(self):
    data_config = _config.DatasetConfig(path="")
    simulator_state = test_utils.make_zeros_state(data_config)

    def is_controlled(state: datatypes.SimulatorState) -> jax.Array:
      return ~state.object_metadata.is_sdc

    actor_core = sim_agent.FrozenSimPolicy(is_controlled)

    actor_state = actor_core.init(jax.random.PRNGKey(0), simulator_state)
    actor_output = actor_core.select_action(
        None, simulator_state, actor_state, jax.random.PRNGKey(0)
    )
    self.assertAllEqual(
        actor_output.is_controlled, ~simulator_state.object_metadata.is_sdc
    )


if __name__ == "__main__":
  tf.test.main()
