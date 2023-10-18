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

"""Manages types specific to Waymax environments used in this library."""
from typing import Callable

import jax

from waymax import datatypes
from waymax import metrics

# Data structures related to simulator observation and the function to compute
# the observation from a given state.
PyTree = datatypes.PyTree
GenericScenario = PyTree
GenericAction = PyTree
GenericState = PyTree
Observation = PyTree
ObservationFn = Callable[[GenericState], Observation]

RewardFn = Callable[[GenericState, GenericAction], jax.Array]

Metrics = dict[str, metrics.MetricResult]
