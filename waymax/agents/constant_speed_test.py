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

import jax.numpy as jnp
import tensorflow as tf

from absl.testing import parameterized
from waymax import config as _config
from waymax import dataloader
from waymax import datatypes
from waymax import dynamics
from waymax.agents import constant_speed
from waymax.utils import test_utils


class ConstantSpeedTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('SpeedZero', 0.0),
      ('SpeedOne', 1.0),
  )
  def test_constant_speed_actor(self, speed):
    max_num_objects = 32
    config = _config.DatasetConfig(
        path=test_utils.ROUTE_DATA_PATH,
        data_format=_config.DataFormat.TFRECORD,
        max_num_objects=max_num_objects,
    )
    state_t0 = next(dataloader.simulator_state_generator(config=config))
    state_t10 = datatypes.update_state_by_log(state_t0, num_steps=10)

    ids = jnp.arange(max_num_objects)
    is_controlled_func = lambda state: ids == 1

    constant_speed_actor = constant_speed.create_constant_speed_actor(
        speed=speed,
        dynamics_model=dynamics.DeltaGlobal(),
        is_controlled_func=is_controlled_func,
    )
    output = constant_speed_actor.select_action(None, state_t10, None, None)  # pytype: disable=wrong-arg-types

    yaw = state_t10.sim_trajectory.yaw[1, state_t10.timestep]
    dx = speed * jnp.cos(yaw) * datatypes.TIME_INTERVAL
    dy = speed * jnp.sin(yaw) * datatypes.TIME_INTERVAL

    self.assertAllClose(output.action.data[1], [dx, dy, 0.0], atol=1e-4)
    self.assertAllClose(output.action.valid[1], [True])


if __name__ == '__main__':
  tf.test.main()
