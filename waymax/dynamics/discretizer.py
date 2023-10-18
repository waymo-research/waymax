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

"""Library for discretizing continuous values and discretizing wrappers."""
import functools

from dm_env import specs
import jax
import jax.numpy as jnp

from waymax import config as _config
from waymax import datatypes
from waymax.dynamics import abstract_dynamics


class Discretizer:
  """Discretizes continuous values into a 1-dimensional array.

  The total number of discrete actions is equal to the product of (all
  bins + 1). We add +1 to be inclusive of boundaries of the min and max values.
  If the continuous value has a shape of (..., 3), and 3 bins are used
  with bin sizes [2, 3, 4], then there will be a total of 60 discrete actions
  (3 * 4 * 5).
  """

  def __init__(
      self, min_value: jax.Array, max_value: jax.Array, bins: jax.Array
  ) -> None:
    """Initializes internal discretizer state.

    Args:
      min_value: Minimal values for the different vector elements to discretize
        of shape (num_vector_elements,).
      max_value: Maximum values for the different vector elements to discretize
        of shape (num_vector_elements,).
      bins: Number of bins for the different vector elements to discretize of
        shape (num_vector_elements,).
    """
    if min_value.shape != max_value.shape and min_value.shape != bins.shape:
      raise ValueError('Shapes do not match.')
    self._mins = min_value
    self._maxs = max_value
    self._bins = bins
    self._shift = min_value
    self._scale = bins / (max_value - min_value)
    self._max_discrete_idx = jnp.prod(self._bins + 1) - 1

  def discretize(self, values: jax.Array) -> jax.Array:
    """Discretizes a continuous batched n-d vector of values to 1d indices.

    Args:
      values: Vector of continuous values of shape (..., num_vector_elements) to
        discretize.

    Returns:
      Discretized values in a tensor of shape (..., 1) with maximum
        value self._max_discrete_idx.
    """
    if values.shape[-1] != self._mins.shape[-1]:
      raise ValueError('Input value shape does not match bin shape.')
    normalized_indices = (values - self._shift) * self._scale
    indices_nd = jnp.rint(
        jnp.maximum(jnp.minimum(normalized_indices, self._bins), 0)
    ).astype(jnp.int32)
    indices_1d = jnp.ravel_multi_index(  # pytype: disable=wrong-arg-types  # jnp-type
        jnp.split(indices_nd, self._bins.shape[0], -1),
        self._bins + 1,
        mode='clip',
    )
    return indices_1d

  def make_continuous(self, indices_1d: jax.Array) -> jax.Array:
    """Takes a discretized matrix and converts it back to continuous values.

    Args:
      indices_1d: Discrete matrix of shape (..., 1) to convert back to
        continuous matrices of shape (..., num_vector_elements).

    Returns:
      Continuous values of shape (..., num_vector_elements) corresponding to the
        value discretized by `indices_1d`.
    """
    indices_nd = jnp.stack(
        jnp.unravel_index(jnp.reshape(indices_1d, [-1]), self._bins + 1),
        axis=-1,
    )
    # Shape: (..., num_vector_elements)
    indices_nd = jnp.reshape(
        indices_nd, list(indices_1d.shape[:-1]) + [self._bins.shape[-1]]
    )
    values = indices_nd.astype(jnp.float32)
    return values / self._scale + self._shift


class DiscreteActionSpaceWrapper(abstract_dynamics.DynamicsModel):
  """Discrete action version of any dynamics model."""

  def __init__(
      self,
      dynamics_model: abstract_dynamics.DynamicsModel,
      bins: jax.Array,
      bin_inverse: bool = True,
  ) -> None:
    """Initializes the bounds of the action space.

    Args:
      dynamics_model: Dynamics model to discretize actions.
      bins: Number of bins for each action space of shape
        dynamics_model.action_spec().shape.
      bin_inverse: Whether to compute continuous inverse and then bin.
        Otherwise, will try all actions and compute min distance over corners.
    """
    super().__init__()
    self._bins = bins
    self._bin_inverse = bin_inverse
    self._num_bins = int(jnp.prod(bins + 1))
    self._cont_dynamics_model = dynamics_model
    self._discretizer = Discretizer(
        min_value=jnp.asarray(self._cont_dynamics_model.action_spec().minimum),
        max_value=jnp.asarray(self._cont_dynamics_model.action_spec().maximum),
        bins=bins,
    )
    self._all_discrete_actions = jnp.arange(self._num_bins).reshape(-1, 1)
    self._all_cont_actions = self._discretizer.make_continuous(
        self._all_discrete_actions
    )
    self._default_cont_action = jnp.zeros(
        (self._cont_dynamics_model.action_spec().shape)
    )
    self._default_discrete_action = self._discretizer.discretize(
        self._default_cont_action
    )

  def action_spec(self) -> specs.BoundedArray:
    """Action spec for discrete dynamics model."""
    return specs.BoundedArray(
        shape=(1,), dtype=jnp.int32, minimum=[0], maximum=[self._num_bins - 1]
    )

  @jax.named_scope('DiscreteActionSpaceWrapper.compute_update')
  def compute_update(
      self,
      action: datatypes.Action,
      trajectory: datatypes.Trajectory,
  ) -> datatypes.TrajectoryUpdate:
    """Computes the pose and velocity updates.

    This function converts the discrete action into a continuous action and
    then runs the underlying continuous action space.

    Args:
      action: Actions to take. Has shape (..., num_objects).
      trajectory: Trajectory to be updated. Has shape of (..., num_objects,
        num_timesteps=1).

    Returns:
      The trajectory update for timestep of shape
        (..., num_objects, num_timesteps=1).
    """
    actions_cont = datatypes.Action(
        data=self._discretizer.make_continuous(action.data),
        valid=action.valid,
    )
    return self._cont_dynamics_model.compute_update(actions_cont, trajectory)

  def inverse(
      self,
      trajectory: datatypes.Trajectory,
      metadata: datatypes.ObjectMetadata,
      timestep: int,
  ) -> datatypes.Action:
    """Calculates the inverse to generate the best fit discrete action.

    If `self._bin_inverse` is False, a discrete optimization algorithm is
    performed to produce the best fit action by searching over the discrete
    action space. If `self._bin_inverse` is True, we discretize the continuous
    inverse. This might be noisier than `self._bin_inverse` = False.

    Args:
      trajectory: A Trajectory used to infer actions of shape (..., num_objects,
        num_timesteps=1).
      metadata: Object metadata for the trajectory of shape (..., num_objects).
      timestep: index of time for actions.

    Returns:
      An Action that converts traj[timestep] to traj[timestep+1] of shape
        (..., num_objects, dim=2).
    """
    if self._bin_inverse:
      action_cont = self._cont_dynamics_model.inverse(
          trajectory, metadata, timestep
      )
      return datatypes.Action(
          data=self._discretizer.discretize(action_cont.data),
          valid=action_cont.valid,
      )
    else:
      inverse_fn = functools.partial(
          inverse_action_by_search,
          continuous_dynamics=self._cont_dynamics_model,
          all_continuous_actions=self._all_cont_actions,
          all_discrete_actions=self._all_discrete_actions,
          discrete_invalid_placeholder=self._default_discrete_action,
      )
      return jax.jit(inverse_fn)(
          trajectory,
          metadata,
          timestep,
      )


def inverse_action_by_search(
    traj: datatypes.Trajectory,
    metadata: datatypes.ObjectMetadata,
    timestep: int,
    continuous_dynamics: abstract_dynamics.DynamicsModel,
    all_continuous_actions: jax.Array,
    all_discrete_actions: jax.Array,
    discrete_invalid_placeholder: jax.Array,
) -> datatypes.Action:
  """Compute the inverse action that best fits a trajectory.

  This inverse method uses a discrete optimization algorithm to produce the
  best fit action by searching over the discrete action space.

  Args:
    traj: A Trajectory used to infer actions of shape (..., num_objects,
      num_timesteps=1).
    metadata: Object metadata for the trajectory of shape (..., num_objects).
    timestep: index of time for actions.
    continuous_dynamics: A continuous-space dynamics model.
    all_continuous_actions: A tensor containing of all possible continuous
      actions of shape (all_actions, dim_action). Each continuous value
      corresponds to one bin of the discretized space.
    all_discrete_actions: A tensor containing the indices of all possible
      discrete actions of shape (all_actions, 1)
    discrete_invalid_placeholder: The default action value of shape (..., 1)
      used as a placeholder for actions that are invalid.

  Returns:
    An Action that converts traj[timestep] to traj[timestep+1] of shape
      (..., num_objects, dim=2).
  """
  is_controlled = datatypes.get_control_mask(metadata, _config.ObjectType.SDC)
  valids = jax.lax.dynamic_slice_in_dim(
      traj.valid, start_index=timestep, slice_size=2, axis=-1
  )
  valid = valids[..., 0:1] & valids[..., 1:2]

  def forward_fn(traj, log_traj, is_controlled, timestep, actions_data):
    return continuous_dynamics.forward(
        datatypes.Action(data=actions_data, valid=valid),
        traj,
        log_traj,
        is_controlled,
        timestep,
    )

  trans_fn = jax.tree_util.Partial(
      forward_fn, traj, traj, is_controlled, timestep
  )
  # Tile to (all_actions, ..., num_objects).
  traj_shape = (
      [all_continuous_actions.shape[0]]
      + [1] * (len(traj.shape) - 1)
      + [all_continuous_actions.shape[1]]
  )
  actions_data = jnp.tile(
      all_continuous_actions.reshape(traj_shape),
      [1] + list(traj.shape[:-1]) + [1],
  )
  # shape: (..., all_actions, num_objects, 1)
  next_traj = datatypes.dynamic_slice(
      jax.vmap(trans_fn)(actions_data),
      start_index=timestep + 1,
      slice_size=1,
      axis=-1,
  )
  # shape: (..., num_objects, 1)
  log_traj = datatypes.dynamic_slice(
      traj, start_index=timestep + 1, slice_size=1, axis=-1
  )
  # Shape: (..., all_actions, num_objects)
  distances = jnp.linalg.norm(
      next_traj.bbox_corners.reshape(*next_traj.shape, -1)
      - log_traj.bbox_corners.reshape(1, *log_traj.shape, -1),
      axis=-1,
  )[..., 0]

  best_action = all_discrete_actions[jnp.argmin(distances, axis=0)]
  # Defaults to 0s for invalid actions.
  action_array = jnp.where(valid, best_action, discrete_invalid_placeholder)
  return datatypes.Action(data=action_array, valid=valid)
