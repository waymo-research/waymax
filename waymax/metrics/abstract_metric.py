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

"""Simulator abstract metric definition."""

import abc

import chex
from jax import numpy as jnp

from waymax import datatypes


@chex.dataclass
class MetricResult(datatypes.MaskedArray):
  """A metric result holding metric values and a validity mask.

  Attributes:
    value: A (...) float32 array of values.
    valid: A (...) boolean validity mask.
  """

  def validate(self) -> None:
    """Validates shape and type."""
    chex.assert_equal_shape([self.value, self.valid])
    chex.assert_type(self.value, jnp.float32)
    chex.assert_type(self.valid, jnp.bool_)


class AbstractMetric(metaclass=abc.ABCMeta):
  """Abstract class for simulator metrics."""

  @abc.abstractmethod
  def compute(self, simulator_state: datatypes.SimulatorState) -> MetricResult:
    """Computes a per-timestep, per-object metric from a SimulatorState.

    Args:
      simulator_state: The current simulator state of shape (...).

    Returns:
      A MetricResult containing the metric result and validity mask of shape
      (..., num_objects).
    """
