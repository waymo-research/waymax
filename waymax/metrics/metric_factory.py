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

"""Utility function that runs all metrics according to an environment config."""
from waymax import config as _config
from waymax import datatypes
from waymax.metrics import abstract_metric
from waymax.metrics import comfort
from waymax.metrics import imitation
from waymax.metrics import overlap
from waymax.metrics import roadgraph
from waymax.metrics import route


def run_metrics(
    simulator_state: datatypes.SimulatorState,
    metrics_config: _config.MetricsConfig,
) -> dict[str, abstract_metric.MetricResult]:
  """Runs all metrics with config flags set to True.

  Args:
    simulator_state: The current simulator state of shape (...).
    metrics_config: Waymax metrics config.

  Returns:
    A dictionary of metric names mapping to metric result arrays where each
      metric is of shape (..., num_objects).
  """
  name_to_metric = {
      'log_divergence': imitation.LogDivergenceMetric,
      'overlap': overlap.OverlapMetric,
      'offroad': roadgraph.OffroadMetric,
      'sdc_wrongway': roadgraph.WrongWayMetric,
      'sdc_progression': route.ProgressionMetric,
      'sdc_off_route': route.OffRouteMetric,
      'sdc_kinematic_infeasibility': comfort.KinematicsInfeasibilityMetric,
  }

  results = {}
  for metric_name, metric_fn in name_to_metric.items():
    # If flag set to True, compute and store metric.
    if getattr(metrics_config, f'run_{metric_name}'):
      results[metric_name] = metric_fn().compute(simulator_state)

  return results
