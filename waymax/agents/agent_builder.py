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

"""Waymax sim agent builder functions."""
from waymax import config as _config
from waymax.agents import actor_core
from waymax.agents import waypoint_following_agent


def create_sim_agents_from_config(
    config: _config.SimAgentConfig,
) -> actor_core.WaymaxActorCore:
  """Constructs sim agent WaymaxActorCore objects from a config.

  Args:
    config: Waymax sim agent config specifying agent type and controlled
      objects' type.

  Returns:
    Constructed sim agents.
  """
  if config.controlled_objects == _config.ObjectType.SDC:
    is_controlled_func = lambda state: state.object_metadata.is_sdc
  elif config.controlled_objects == _config.ObjectType.NON_SDC:
    is_controlled_func = lambda state: ~state.object_metadata.is_sdc
  else:
    raise NotImplementedError(
        f'{config.controlled_objects} is not a supported controlled objects for'
        ' sim agent.'
    )

  if config.agent_type == _config.SimAgentType.IDM:
    return waypoint_following_agent.IDMRoutePolicy(is_controlled_func)
  else:
    raise NotImplementedError(f'Agent {config.agent_type} is not supported.')
