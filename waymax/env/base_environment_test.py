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

"""Tests for env."""

import dataclasses

import jax
from jax import numpy as jnp
import tensorflow as tf

from absl.testing import parameterized
from waymax import config as _config
from waymax import dataloader
from waymax import datatypes
from waymax import dynamics
from waymax.env import base_environment as _env
from waymax.utils import test_utils


TEST_DATA_PATH = test_utils.ROUTE_DATA_PATH


class BaseEnvironmentTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    # TODO(b/251258357) Update environment tests to test different inputs args
    # including dynamics and controlled objects.
    self.env_config = _config.EnvironmentConfig(init_steps=10)
    self.env = _env.MultiAgentEnvironment(
        dynamics_model=dynamics.DeltaGlobal(), config=self.env_config
    )
    action_spec = self.env.action_spec()
    self.sample_action = datatypes.Action(
        data=jnp.zeros(action_spec.data.shape, dtype=jnp.float32),
        valid=jnp.ones(action_spec.valid.shape, dtype=jnp.bool_),
    )

    dataset_config = _config.DatasetConfig(
        path=TEST_DATA_PATH, data_format=_config.DataFormat.TFRECORD
    )
    dataset_iter = dataloader.simulator_state_generator(dataset_config)
    self.sample_state = next(dataset_iter)

  def test_reset_matches_history_values(self):
    reset_state = self.env.reset(self.sample_state)
    init_steps = self.env.config.init_steps
    log_slice = datatypes.dynamic_slice(
        self.sample_state.log_trajectory, 0, init_steps, axis=-1
    )
    new_traj = datatypes.dynamic_update_slice_in_dim(
        self.sample_state.sim_trajectory, log_slice, 0, axis=-1
    )
    expected_state = self.sample_state.replace(
        sim_trajectory=new_traj, timestep=init_steps - 1
    )
    self.assertAllClose(
        reset_state.sim_trajectory.xyz[..., :init_steps, :],
        expected_state.sim_trajectory.xyz[..., :init_steps, :],
    )
    self.assertEqual(reset_state.timestep, init_steps - 1)

  def test_reset_produces_correct_values(self):
    # Set up a test where the log trajectory contains 0s, and the sim
    # trajectory is initialized with 1s. The reset should invalidate
    # all entries in the sim trajectory and fill the beginning of
    # the log trajectory with 0s.
    num_agents = 2
    num_roadgraph_points = 2000
    num_tls = 3
    num_timesteps = 10
    traj_value = jnp.ones((num_agents, num_timesteps))
    sim_traj = datatypes.Trajectory(
        x=traj_value,
        y=traj_value,
        z=traj_value,
        vel_x=traj_value,
        vel_y=traj_value,
        yaw=traj_value,
        length=traj_value,
        width=traj_value,
        height=traj_value,
        timestamp_micros=jnp.ones_like(traj_value).astype(jnp.int32),
        valid=jnp.ones_like(traj_value).astype(jnp.bool_),
    )
    log_traj = jax.tree.map(
        lambda x: jnp.zeros_like(x).astype(x.dtype), sim_traj
    )
    roadgraph_points = datatypes.RoadgraphPoints(
        x=jnp.zeros((num_roadgraph_points,)),
        y=jnp.zeros((num_roadgraph_points,)),
        z=jnp.zeros((num_roadgraph_points,)),
        dir_x=jnp.zeros((num_roadgraph_points,)),
        dir_y=jnp.zeros((num_roadgraph_points,)),
        dir_z=jnp.zeros((num_roadgraph_points,)),
        types=jnp.zeros((num_roadgraph_points,), dtype=jnp.int32),
        ids=jnp.zeros((num_roadgraph_points,), dtype=jnp.int32),
        valid=jnp.zeros((num_roadgraph_points,), dtype=jnp.bool_),
    )
    traffic_lights = datatypes.TrafficLights(
        x=jnp.zeros((num_tls, num_timesteps)),
        y=jnp.zeros((num_tls, num_timesteps)),
        z=jnp.zeros((num_tls, num_timesteps)),
        state=jnp.zeros((num_tls, num_timesteps), dtype=jnp.int32),
        lane_ids=jnp.zeros((num_tls, num_timesteps), dtype=jnp.int32),
        valid=jnp.zeros((num_tls, num_timesteps), dtype=jnp.bool_),
    )
    metadata = datatypes.ObjectMetadata(
        ids=jnp.ones((num_agents,), dtype=jnp.int32),
        object_types=jnp.ones((num_agents,), dtype=jnp.int32),
        is_sdc=jnp.zeros((num_agents,), dtype=jnp.bool_),
        is_modeled=jnp.zeros((num_agents,), dtype=jnp.bool_),
        is_valid=jnp.ones((num_agents,), dtype=jnp.bool_),
        objects_of_interest=jnp.zeros((num_agents,), dtype=jnp.bool_),
        is_controlled=jnp.zeros((num_agents,), dtype=jnp.bool_),
    )
    sample_state = datatypes.SimulatorState(
        sim_trajectory=sim_traj,
        log_trajectory=log_traj,
        log_traffic_light=traffic_lights,
        object_metadata=metadata,
        roadgraph_points=roadgraph_points,
        timestep=0,
    )
    env = _env.MultiAgentEnvironment(
        dynamics_model=dynamics.DeltaGlobal(),
        config=_config.EnvironmentConfig(
            init_steps=5, max_num_objects=num_agents
        ),
    )
    reset_state = env.reset(sample_state)

    # Since we reset with 5 steps, the first 5 entries should be 0 from the
    # log trajectory, with the remainder as -1 marking invalid.
    expected_value = jnp.array(
        [0.0, 0.0, 0.0, 0.0, 0.0, -1.0, -1.0, -1.0, -1.0, -1.0]
    )
    self.assertAllClose(reset_state.sim_trajectory.x[0, :], expected_value)
    self.assertAllClose(reset_state.sim_trajectory.y[0, :], expected_value)
    self.assertAllClose(reset_state.sim_trajectory.z[0, :], expected_value)

  def test_initial_state_sets_initial_timestep(self):
    init_state = self.env.reset(self.sample_state)
    self.assertEqual(init_state.timestep, self.env_config.init_steps - 1)

  def test_transition_advances_timestep(self):
    init_state = self.env.reset(self.sample_state)
    next_state = self.env.step(init_state, self.sample_action)
    self.assertEqual(next_state.timestep, init_state.timestep + 1)

  @parameterized.product(
      object_type=(
          _config.ObjectType.SDC,
          _config.ObjectType.MODELED,
          _config.ObjectType.VALID,
      ),
      allow_new_objects_after_warmup=(True, False),
  )
  def test_get_is_controlled_works_correctly(
      self, object_type, allow_new_objects_after_warmup
  ):
    reset_state = self.env.reset(self.sample_state)
    env_config = dataclasses.replace(
        self.env.config,
        controlled_object=object_type,
        allow_new_objects_after_warmup=allow_new_objects_after_warmup,
    )
    if object_type == _config.ObjectType.SDC:
      expected_value = reset_state.object_metadata.is_sdc
    elif object_type == _config.ObjectType.MODELED:
      expected_value = reset_state.object_metadata.is_modeled
    elif object_type == _config.ObjectType.VALID:
      if allow_new_objects_after_warmup:
        expected_value = reset_state.object_metadata.is_valid
      else:
        expected_value = reset_state.current_sim_trajectory.valid[..., 0]
    else:
      raise ValueError(f'{object_type} not supported.')
    self.assertAllEqual(
        _env._get_control_mask(reset_state, env_config),
        expected_value,
    )


if __name__ == '__main__':
  tf.test.main()
