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

"""Waymax data loading functions.

WOMD represents the data we typically use for simulation in our environment. See
https://waymo.com/open/data/motion/tfexample for definitions on many of the data
fields mentioned in this file.
"""

import functools
from typing import Iterator, Optional, TypeVar

import jax.numpy as jnp
import tensorflow as tf

from waymax import config as _config
from waymax.dataloader import dataloader_utils
from waymax.dataloader import womd_factories
from waymax.dataloader import womd_utils
from waymax.datatypes import simulator_state


T = TypeVar('T')
AUTOTUNE = tf.data.AUTOTUNE


def preprocess_serialized_womd_data(
    serialized: bytes, config: _config.DatasetConfig
) -> dict[str, tf.Tensor]:
  """Parses serialized tf example into tf Tensor dict."""
  womd_features = womd_utils.get_features_description(
      include_sdc_paths=config.include_sdc_paths,
      max_num_rg_points=config.max_num_rg_points,
      num_paths=config.num_paths,
      num_points_per_path=config.num_points_per_path,
  )

  deserialized = tf.io.parse_example(serialized, womd_features)
  return preprocess_womd_example(
      deserialized,
      aggregate_timesteps=config.aggregate_timesteps,
      max_num_objects=config.max_num_objects,
  )


def preprocess_womd_example(
    example: dict[str, tf.Tensor],
    aggregate_timesteps: bool,
    max_num_objects: Optional[int] = None,
) -> dict[str, tf.Tensor]:
  """Preprocesses dict of tf tensors, keyed by str."""

  if aggregate_timesteps:
    processed = womd_utils.aggregate_time_tensors(example)
    wrap_yaws = lambda yaws: (yaws + jnp.pi) % (2 * jnp.pi) - jnp.pi
    processed['state/all/bbox_yaw'] = wrap_yaws(processed['state/all/bbox_yaw'])
  else:
    processed = example

  if max_num_objects is not None:
    # TODO check sdc included if it is needed.
    return {
        k: v[:max_num_objects] if k.startswith('state/') else v
        for k, v in processed.items()
    }
  else:
    return processed


def simulator_state_generator(
    config: _config.DatasetConfig,
) -> Iterator[simulator_state.SimulatorState]:
  """Wrapper for SimulatorState iterator.

  This is the high level api for Waymax data loading that takes Waymax data
  config and outputs generator of SimulatorState.

  Args:
    config: dataset config.

  Returns:
    A SimulatorState iterator.
  """

  parse_and_preprocess = functools.partial(
      preprocess_serialized_womd_data, config=config
  )
  womd_dict_to_sim_state_func = functools.partial(
      womd_factories.simulator_state_from_womd_dict,
      include_sdc_paths=config.include_sdc_paths,
  )


  return dataloader_utils.get_data_generator(
      config, parse_and_preprocess, womd_dict_to_sim_state_func
  )
