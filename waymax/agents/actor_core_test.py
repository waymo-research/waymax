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

from typing import Any

import jax
import jax.numpy as jnp
import tensorflow as tf

from waymax import config as _config
from waymax import datatypes
from waymax.agents import actor_core
from waymax.utils import test_utils


@jax.jit
def jitted_act_function(
    actor: actor_core.WaymaxActorCore, simulator_state: datatypes.SimulatorState
) -> actor_core.WaymaxActorOutput:
  actor_state = actor.init(jax.random.PRNGKey(0), simulator_state)
  return actor.select_action(
      None, simulator_state, actor_state, jax.random.PRNGKey(0)
  )


class ActorCoreTest(tf.test.TestCase):

  def test_actor_core_is_jittable(self):
    data_config = _config.DatasetConfig(path="")
    simulator_state = test_utils.make_zeros_state(data_config)

    def select_action(
        params: Any,
        state: datatypes.SimulatorState,
        actor_state: Any,
        key: jax.Array,
    ) -> actor_core.WaymaxActorOutput:
      del params, actor_state, key  # Unused
      action = datatypes.TrajectoryUpdate(
          x=state.sim_trajectory.x,
          y=state.sim_trajectory.y,
          yaw=state.sim_trajectory.yaw,
          vel_x=state.sim_trajectory.vel_x,
          vel_y=state.sim_trajectory.vel_y,
          valid=state.sim_trajectory.valid,
      ).as_action()
      return actor_core.WaymaxActorOutput(
          actor_state=None,
          action=action,
          is_controlled=state.object_metadata.is_sdc,
      )

    actor = actor_core.actor_core_factory(
        init=lambda key, state: None,
        select_action=select_action,
        name="TestActorCore",
    )
    # Running jitted function should not throw an error.
    jitted_act_function(actor, simulator_state)

  def test_merge_actions(self):
    action_a = datatypes.TrajectoryUpdate(
        x=jnp.array([[0.0], [0.0], [0.0]]),
        y=jnp.array([[1.0], [1.0], [1.0]]),
        yaw=jnp.array([[1.0], [1.0], [1.0]]),
        vel_x=jnp.array([[0.0], [0.0], [0.0]]),
        vel_y=jnp.array([[1.0], [1.0], [1.0]]),
        valid=jnp.array([[True], [False], [False]]),
    ).as_action()
    output_a = actor_core.WaymaxActorOutput(
        actor_state=None,
        action=action_a,
        is_controlled=jnp.array([True, False, False]),
    )

    action_b = datatypes.TrajectoryUpdate(
        x=jnp.array([[1.0], [1.0], [1.0]]),
        y=jnp.array([[0.0], [0.0], [0.0]]),
        yaw=jnp.array([[2.0], [2.0], [2.0]]),
        vel_x=jnp.array([[1.0], [1.0], [1.0]]),
        vel_y=jnp.array([[0.0], [0.0], [0.0]]),
        valid=jnp.array([[False], [True], [True]]),
    ).as_action()
    output_b = actor_core.WaymaxActorOutput(
        actor_state=None,
        action=action_b,
        is_controlled=jnp.array([False, True, True]),
    )

    expected = datatypes.TrajectoryUpdate(
        x=jnp.array([[0.0], [1.0], [1.0]]),
        y=jnp.array([[1.0], [0.0], [0.0]]),
        yaw=jnp.array([[1.0], [2.0], [2.0]]),
        vel_x=jnp.array([[0.0], [1.0], [1.0]]),
        vel_y=jnp.array([[1.0], [0.0], [0.0]]),
        valid=jnp.array([[True], [True], [True]]),
    ).as_action()

    action = actor_core.merge_actions([output_a, output_b])
    self.assertEqual(action, expected)


if __name__ == "__main__":
  tf.test.main()
