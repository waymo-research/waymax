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

"""Utilities for fast jittable rollout of environments in Waymax."""

from typing import Optional

import chex
import jax
from jax import numpy as jnp
from waymax import dynamics
from waymax.agents import actor_core
from waymax.agents import expert
from waymax.env import abstract_environment
from waymax.env import typedefs as types


@chex.dataclass(frozen=True)
class RolloutCarry:
  """Rollout output that is carried between iterations of the rollout.

  Attributes:
    state: Current state of the simulator after `state.timestep` steps.
    observation: Observation of the simulator state from the environment which
      is called by a given observation function meant to be the input of the
      actor's model.
    rng: Random key which represents the key for randomizing actions and
      initializing parameters for a Waymax actor.
    actor_state: Internal state for whatever the agent needs to keep as its
      state. This can be recurrent embeddings or accounting information.
  """

  state: types.GenericState
  observation: types.Observation
  rng: jax.Array
  actor_state: types.PyTree


@chex.dataclass(frozen=True)
class RolloutOutput:
  """Rollout output datatypes.structure for using as output of rollout function.

  Attributes:
    action: Action produced by a functional corresponding to `ActionFuncType`
      which after taking by calling some `environment.step(action)` produces the
      `timestep` information. This is aggregated over a number of timesteps and
      so the shape is (num_timesteps, ..., num_objects, action_dim). The `...`
      of the shapes correspond to any kind of prefix for batching that might be
      applied.
    state: Temporally aggregated information of the output of the simulation
      after calling `environment.step(action)`. This information represents the
      important information from the simulation aggregated through the rollout
      of shape (num_timesteps, ...). The first element of `state` corresponds to
      the initial simulation state.
    observation: Temporally aggregated information of the output of the
      simulation after calling `observe(environment.step(action))`. This
      information  represents the observation of the agent of the simulator
      state aggregated through the rollout of shape (num_timesteps, ...). The
      first element of `observation` corresponds to the initial simulation
      state.
    metrics: Mapping from metric name to metric which contains metrics computed
      on the simulator states aggregated in time of shape (num_timestpes, ...).
      These functions are defined in the `env.metrics(state)` function. As this
      is a mapping, these metrics could be empty if the environment decides not
      to produce metrics. This could be due to speed reasons during the rollout.
    reward: Scalar value of shape (num_timesteps, ..., num_objects) which
      represents the reward achieved at a certain simulator state at the given
      `state.timestep`.
  """

  action: types.GenericAction
  state: types.GenericState
  observation: types.Observation
  metrics: types.Metrics
  reward: jax.Array

  def validate(self):
    """Validates the shape prefix of the actions and timesteps."""
    chex.assert_equal_shape_prefix(
        (self.action, self.observation, self.state, self.reward),
        prefix_len=len(self.state.shape),
    )

  @property
  def shape(self) -> tuple[int, ...]:
    """Returns the shape prefix for the rollout type."""
    return self.state.shape


# TODO(b/251256348): Update tests for rollout function.
@jax.named_scope('rollout')
def rollout(
    scenario: types.GenericScenario,
    actor: actor_core.WaymaxActorCore,
    env: abstract_environment.AbstractEnvironment,
    rng: jax.Array,
    rollout_num_steps: int = 1,
    actor_params: Optional[actor_core.Params] = None,
) -> RolloutOutput:
  """Performs a rollout from the beginning of a run segment.


  Args:
    scenario: initial SimulatorState to start the rollout of shape (...).
    actor: The action function used to select actions during the rollout.
    env: A stateless Waymax environment used for computing steps, observations,
      and rewards.
    rng: Random key used for generate stochastic actions if needed.
    rollout_num_steps: number of rollout steps.
    actor_params: Parameters used by actor to select actions. It can be None if
      the actor does not require parameters.

  Returns:
    Stacked rollout output  of shape (rollout_num_steps + 1, ...) from the
      simulator when taking an action given the action_fn. There is one extra in
      the time dimension compared to `rollout_num_steps`. This is because we
      prepend the initial timetep to the `timestep` field and append an invalid
      action into the `action` field.
  """
  # TODO(b/246965197) Runtime check that rollout_num_steps is not greater than
  # reset_state.remaining_timesteps
  reset_state = env.reset(scenario)
  init_rng, rng = jax.random.split(rng, 2)
  actor_init_state = actor.init(init_rng, reset_state)

  def _step(
      carry: RolloutCarry, timestep: int
  ) -> tuple[RolloutCarry, RolloutOutput]:
    """Runs one step of the rollout.

    One step of the rollout consists of using the `action_fn` to return an
    action which is used to step through the environment one time.

    Args:
      carry: Output from the previous iteration of the simulation that carries
        over into this iteration. This will be the first element of the tuple
        returned by this function.
      timestep: Current timestep of the simulation.

    Returns:
      Output of this step of simulation. The first element of the tuple
        represents the part of the output carried over into the next step of
        simulation while the second element of the tuple is the final output of
        the simulation which is stacked in the first element.
    """
    del timestep
    action_rng, rng = jax.random.split(carry.rng, 2)
    actor_output = actor.select_action(
        actor_params, carry.state, carry.actor_state, action_rng
    )
    next_state = env.step(carry.state, actor_output.action)
    next_observation = env.observe(next_state)
    next_carry = RolloutCarry(
        state=next_state,
        observation=next_observation,
        rng=rng,
        actor_state=actor_output.actor_state,
    )
    return next_carry, RolloutOutput(
        action=actor_output.action,
        state=carry.state,
        observation=carry.observation,
        metrics=env.metrics(carry.state),
        reward=env.reward(carry.state, actor_output.action),
    )

  init_carry = RolloutCarry(
      state=reset_state,
      observation=env.observe(reset_state),
      rng=rng,
      actor_state=actor_init_state,
  )
  carry, output = jax.lax.scan(
      _step, init_carry, xs=jnp.arange(rollout_num_steps)
  )

  padding_action = jax.tree_util.tree_map(
      lambda x: jnp.zeros_like(x[-1]), output.action
  )
  last_output = RolloutOutput(
      action=padding_action,
      state=carry.state,
      observation=carry.observation,
      metrics=env.metrics(carry.state),
      reward=env.reward(carry.state, padding_action),
  )

  output = jax.tree_util.tree_map(
      lambda x, y: jnp.concatenate([x, y[jnp.newaxis]], axis=0),
      output,
      last_output,
  )
  output.validate()
  return output


def rollout_log_by_expert_sdc(
    scenario: types.GenericScenario,
    env: abstract_environment.AbstractEnvironment,
    dynamics_model: dynamics.DynamicsModel,
    rollout_num_steps: int = 1,
) -> RolloutOutput:
  """Rollouts state using logged expert actions specified by dynamics_model."""
  return rollout(
      scenario,
      expert.create_expert_actor(dynamics_model),
      env,
      rng=jax.random.PRNGKey(0),  # Not used for expert actions.
      rollout_num_steps=rollout_num_steps,
  )
