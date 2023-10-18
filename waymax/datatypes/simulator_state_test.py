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

"""Tests for data_structures."""

from jax import numpy as jnp
import tensorflow as tf

from absl.testing import parameterized
from waymax.dataloader import womd_factories
from waymax.utils import test_utils


class DataStructTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters(((), 27), ((2, 1), 53))
  def test_get_current_sim_trajectory(self, batch_dims, timestep):
    dataset = test_utils.make_test_dataset(batch_dims=batch_dims)
    data_dict = next(dataset.as_numpy_iterator())
    sim_state = womd_factories.simulator_state_from_womd_dict(
        data_dict, time_key='all'
    )

    timestamps = jnp.tile(jnp.arange(91), batch_dims + (128, 1))
    new_traj = sim_state.sim_trajectory.replace(timestamp_micros=timestamps)
    sim_state = sim_state.replace(sim_trajectory=new_traj, timestep=timestep)
    current_traj = sim_state.current_sim_trajectory
    # Shape is batch dims + (num_objects, timesteps)
    self.assertEqual(current_traj.shape, batch_dims + (128, 1))
    # Check that the right timestep got selected.
    self.assertAllClose(
        current_traj.timestamp_micros,
        jnp.ones(batch_dims + (128, 1)) * timestep,
    )


if __name__ == '__main__':
  tf.test.main()
