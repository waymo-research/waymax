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

"""Constant speed agents."""

from typing import Callable, Optional

import jax
import jax.numpy as jnp

from waymax import datatypes
from waymax import dynamics
from waymax.agents import actor_core
from waymax.agents import waypoint_following_agent


def create_constant_speed_actor(
    dynamics_model: dynamics.DynamicsModel,
    is_controlled_func: Callable[[datatypes.SimulatorState], jax.Array],
    speed: Optional[float] = None,
) -> actor_core.WaymaxActorCore:
  """Creates an actor with constant speed without changing objects' heading.

  Note the difference against ConstantSpeedPolicy is that an actor requires
  input of a dynamics model, while a policy does not (it assumes to use
  StateDynamics).

  Args:
    dynamics_model: The dynamics model the actor is using that defines the
      action output by the actor.
    is_controlled_func: Defines which objects are controlled by this actor.
    speed: Speed of the actor, if None, speed from previous step is used.

  Returns:
    An statelss actor that drives the controlled objects with constant speed.
  """

  def select_action(  # pytype: disable=annotation-type-mismatch
      params: actor_core.Params,
      state: datatypes.SimulatorState,
      actor_state=None,
      rng: jax.Array = None,
  ) -> actor_core.WaymaxActorOutput:
    """Computes the actions using the given dynamics model and speed."""
    del params, actor_state, rng  # unused.
    traj_t0 = datatypes.dynamic_index(
        state.sim_trajectory, state.timestep, axis=-1, keepdims=True
    )
    if speed is None:
      vel_x = traj_t0.vel_x
      vel_y = traj_t0.vel_y
    else:
      vel_x = speed * jnp.cos(traj_t0.yaw)
      vel_y = speed * jnp.sin(traj_t0.yaw)

    is_controlled = is_controlled_func(state)
    traj_t1 = traj_t0.replace(
        x=traj_t0.x + vel_x * datatypes.TIME_INTERVAL,
        y=traj_t0.y + vel_y * datatypes.TIME_INTERVAL,
        vel_x=vel_x,
        vel_y=vel_y,
        valid=is_controlled[..., jnp.newaxis] & traj_t0.valid,
        timestamp_micros=(
            traj_t0.timestamp_micros + datatypes.TIMESTEP_MICROS_INTERVAL
        ),
    )

    traj_combined = jax.tree_util.tree_map(
        lambda x, y: jnp.concatenate((x, y), axis=-1), traj_t0, traj_t1
    )
    actions = dynamics_model.inverse(
        traj_combined, state.object_metadata, timestep=0
    )

    # Note here actions' valid could be different from is_controlled, it happens
    # when that object does not have valid trajectory from the previous
    # timestep.
    return actor_core.WaymaxActorOutput(
        actor_state=None,
        action=actions,
        is_controlled=is_controlled,
    )

  return actor_core.actor_core_factory(
      init=lambda rng, init_state: None,
      select_action=select_action,
      name=f'constant_speed_{speed}',
  )


class ConstantSpeedPolicy(waypoint_following_agent.WaypointFollowingPolicy):
  """A policy that maintains a constant speed for all sim agents."""

  def __init__(self, speed: float = 0.0):
    """Creates a ConstantSpeedPolicy.

    Args:
      speed: Speed in m/s to set as the speed for all agents.
    """
    super().__init__(invalidate_on_end=True)
    self._speed = speed

  def update_speed(
      self, state: datatypes.SimulatorState, dt: float = 0.1
  ) -> tuple[jax.Array, jax.Array]:
    """Sets the speed for each agent in the current sim step to a constant.

    Args:
      state: The simulator state of shape (...).
      dt: Delta between timesteps of the simulator state.

    Returns:
      speeds: A (..., num_objects) float array of constant speeds.
      valids: A (..., num_objects) bool array of valids.
    """
    shape = state.shape + (state.num_objects,)
    return (jnp.full(shape, self._speed), jnp.ones(shape, dtype=jnp.bool_))
