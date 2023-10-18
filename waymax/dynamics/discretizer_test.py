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
from waymax import datatypes
from waymax.dataloader import womd_dataloader
from waymax.dynamics import delta
from waymax.dynamics import discretizer as _discretizer
from waymax.utils import test_utils

TEST_DATA_PATH = test_utils.ROUTE_DATA_PATH


class DiscretizerTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.sim_state = next(
        womd_dataloader.simulator_state_generator(
            config=_config.DatasetConfig(
                path=TEST_DATA_PATH,
                max_num_objects=32,
                data_format=_config.DataFormat.TFRECORD,
            )
        )
    )

  def test_min_bounds(self):
    """Test actions can map to min bounds."""
    min_bounds = jnp.array([-3, -3])
    max_bounds = jnp.array([3, 3])
    bins = jnp.array([7, 31])
    discretizer = _discretizer.Discretizer(min_bounds, max_bounds, bins)
    self.assertEqual(discretizer.discretize(min_bounds), jnp.array([0]))
    self.assertAllClose(discretizer.make_continuous(jnp.array([0])), min_bounds)

  def test_max_bounds(self):
    """Test actions can map to max bounds."""
    min_bounds = jnp.array([-3, -3])
    max_bounds = jnp.array([3, 3])
    bins = jnp.array([7, 31])
    discretizer = _discretizer.Discretizer(min_bounds, max_bounds, bins)
    # The number of discrete action bins is the product of all bin sizes +1,
    # in order to include the boundary values as bins.
    max_discrete_index = 255  # Computed as jnp.prod(bins + 1) - 1
    self.assertEqual(
        discretizer.discretize(values=max_bounds), max_discrete_index
    )
    self.assertAllClose(
        discretizer.make_continuous(jnp.array([max_discrete_index])), max_bounds
    )

  def test_mid_bounds(self):
    """Test actions can map to mid bounds."""
    min_bounds = jnp.array([-3, -3])
    max_bounds = jnp.array([3, 3])
    bins = jnp.array([6, 30])
    discretizer = _discretizer.Discretizer(min_bounds, max_bounds, bins)
    self.assertEqual(discretizer.discretize(jnp.array([0, 0])), 108)
    self.assertAllClose(
        discretizer.make_continuous(jnp.array([108])), jnp.array([0, 0])
    )

  def test_batch_action(self):
    """Test discretizer works on batch of actions."""
    min_bounds = jnp.array([-3, -3])
    max_bounds = jnp.array([3, 3])
    bins = jnp.array([7, 31])
    discretizer = _discretizer.Discretizer(min_bounds, max_bounds, bins)
    max_discrete = jnp.prod(bins + 1) - 1
    cont_actions = jnp.zeros((2, 1)) + max_bounds
    discrete_actions = jnp.array([[max_discrete], [max_discrete]])
    self.assertAllClose(discretizer.discretize(cont_actions), discrete_actions)
    self.assertAllClose(
        discretizer.make_continuous(discrete_actions), cont_actions
    )

  def test_round_trip(self):
    min_bounds = jnp.array([-3, -3])
    max_bounds = jnp.array([3, 3])
    bins = jnp.array([6, 30])
    discretizer = _discretizer.Discretizer(min_bounds, max_bounds, bins)
    cont_action = jnp.array([0, 0])
    self.assertAllClose(
        discretizer.make_continuous(discretizer.discretize(cont_action)),
        cont_action,
    )

  def test_all_discrete_map_unique(self):
    min_bounds = jnp.array([-3, -3])
    max_bounds = jnp.array([3, 3])
    bins = jnp.array([6, 30])
    discretizer = _discretizer.Discretizer(min_bounds, max_bounds, bins)
    all_discrete = jnp.arange(jnp.prod(bins + 1)).reshape(-1, 1)
    all_continuous = discretizer.make_continuous(all_discrete)
    self.assertAllClose(all_discrete, discretizer.discretize(all_continuous))

  @parameterized.product(
      batch_dims=((), (2, 1)),
      which_model=('DeltaGlobal', 'DeltaLocal'),
      controlled_obj=(
          _config.ObjectType.SDC,
          _config.ObjectType.MODELED,
      ),
      bin_inverse=(True, False),
  )
  def test_discrete_dynamics_model_recovers_log_by_inverse_and_forward(
      self, batch_dims, which_model, controlled_obj, bin_inverse
  ):
    # This test checks that forward and inverse are inverses of each other.
    # We first infer an action using inverse dynamics, and run the forward
    # update. We check that the forward update matches the original trajectory.
    # Read the initial state at t=0 from the dataset.
    state_t0 = next(
        womd_dataloader.simulator_state_generator(
            config=_config.DatasetConfig(
                path=TEST_DATA_PATH,
                max_num_objects=32,
                batch_dims=batch_dims,
                data_format=_config.DataFormat.TFRECORD,
            )
        )
    )
    # Initialize the first 10 steps of sim_trajectory with log_trajectory.
    state_t10 = datatypes.update_state_by_log(state_t0, num_steps=10)
    is_controlled = datatypes.get_control_mask(
        state_t0.object_metadata, controlled_obj
    )
    bins = jnp.array([50, 50, 30])
    cont_dynamics_model = getattr(delta, which_model)()
    discrete_dynamics_model = _discretizer.DiscreteActionSpaceWrapper(
        cont_dynamics_model, bins, bin_inverse=bin_inverse
    )
    # Run inverse dynamics on logged trajectory to infer actions.
    actions = discrete_dynamics_model.inverse(
        state_t10.log_trajectory, state_t0.object_metadata, timestep=10
    )
    # Apply inferred actions to sim trajectory to recover logged trajectory.
    out = discrete_dynamics_model.forward(
        actions,
        trajectory=state_t10.sim_trajectory,
        reference_trajectory=state_t10.log_trajectory,
        is_controlled=is_controlled,
        timestep=10,
    )
    action_spec = discrete_dynamics_model.action_spec()
    tolerance = (action_spec.maximum - action_spec.minimum) / bins
    self.assertAllClose(out.x[..., :11], state_t10.log_trajectory.x[..., :11])
    self.assertAllClose(
        out.x[..., 12:],
        jnp.zeros_like(out.x[..., 12:], dtype=jnp.float32),
        atol=tolerance[0],
    )
    # Recovers for valid (both controlled and un-controlled).
    self.assertAllClose(
        out.x[actions.valid[..., 0], 11],
        state_t10.log_trajectory.x[actions.valid[..., 0], 11],
        atol=tolerance[0],
    )
    self.assertAllClose(
        out.y[actions.valid[..., 0], 11],
        state_t10.log_trajectory.y[actions.valid[..., 0], 11],
        atol=tolerance[1],
    )
    self.assertAllClose(
        out.yaw[actions.valid[..., 0], 11],
        state_t10.log_trajectory.yaw[actions.valid[..., 0], 11],
        atol=tolerance[2],
    )

  @parameterized.product(which_model=('DeltaGlobal', 'DeltaLocal'))
  def test_discrete_inverse_recovers_default_action(self, which_model):
    log_traj = datatypes.Trajectory(
        x=jnp.zeros((2, 10)),
        y=jnp.zeros((2, 10)),
        z=jnp.zeros((2, 10)),
        vel_x=jnp.zeros((2, 10)),
        vel_y=jnp.zeros((2, 10)),
        yaw=jnp.zeros((2, 10)),
        valid=jnp.ones((2, 10), bool),
        timestamp_micros=jnp.zeros((2, 10)),
        length=jnp.ones((2, 10)),
        width=jnp.ones((2, 10)),
        height=jnp.zeros((2, 10)),
    )

    cont_dynamics_model = getattr(delta, which_model)()
    bins = jnp.full(cont_dynamics_model.action_spec().shape, 10)
    discrete_dynamics_model = _discretizer.DiscreteActionSpaceWrapper(
        cont_dynamics_model, bins
    )
    obj_metadata = datatypes.ObjectMetadata(
        ids=jnp.zeros(2),
        object_types=jnp.ones(2),
        is_sdc=jnp.array([1, 0]),
        is_modeled=jnp.ones(2),
        is_valid=jnp.ones(2),
        objects_of_interest=jnp.ones(2),
        is_controlled=jnp.ones(2),
    )
    actions = discrete_dynamics_model.inverse(
        log_traj, obj_metadata, timestep=5
    )
    self.assertAllClose(
        actions.data[0], discrete_dynamics_model._default_discrete_action
    )


if __name__ == '__main__':
  tf.test.main()
