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

"""Configs for Waymax Environments."""
import dataclasses
import enum
from typing import Optional, Sequence


class CoordinateFrame(enum.Enum):
  """Coordinate system used for data."""

  # All objects are centered at SDC's xy and use SDC's yaw as x-axis.
  SDC = 'SDC'
  # Centered at each object's xy and use their yaws as x-axis.
  OBJECT = 'OBJECT'
  # Uses global coordinates defined in Waymo Open Motion Dataset (WOMD).
  GLOBAL = 'GLOBAL'


class DataFormat(enum.Enum):
  """Data format for serialized records."""

  TFRECORD = 'TFRECORD'


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
  """Configs for dataset/dataloader.

  Attributes:
    path: Path/pattern for data, supporting sharded files with @.
    data_format: The format of data in `path`, string from
      womd_dataloader.DataFormat.
    repeat: Number of times to repeat the dataset. Set None to repeat
      infinitely.
    batch_dims: Batch dimensions in tuple format. Can be empty as ().
    shuffle_seed: Random seed. Set None to disable shuffle.
    shuffle_buffer_size: Buffer size for shuffling.
    num_shards: Number of shards for parallel loading, no effect on data
      returned.
    deterministic: Whether to use deterministic parallel processing.
    include_sdc_paths: Whether to include all valid future paths for SDC
      according to roadgraph connectivity from its starting position.
    aggregate_timesteps: Whether to aggregate keys from tf examples, need to set
      to True for parsing SimulatorState afterwards.
    max_num_rg_points: Max number of roadgraph points in data.
    max_num_objects: Max number of total objects in the scene. Set None to
      include all objects from the data.
    num_paths: Number of roadgraph traversal paths. Must be specified if
      include_sdc_paths is True.
    num_points_per_path: Number of points per roadgraph traversal path. Must be
      specified if include_sdc_paths is True.
    drop_remainder: Argument for tf.data.Dataset.batch. Set True to drop
      remainder if the last batch does not contain enough examples. Note
      training should not be affected since it is looping over all data for
      multiple epochs. For evaluation, it should be set to False to include all
      examples.
    tf_data_service_address: A string or a tuple indicating how to connect to
      the tf.data service. Used as the `service` argument for
      tf.data.experimental.service.distribute in dataloader.
    distributed: If True, the generated data batch will have an extra leading
      axis corresponding to the number of available devices. This is needed when
      the returned data is consumed by a pmap function.
    batch_by_scenario: If True, one example in a returned batch is the entire
      scenario containing all objects; if False, the dataset will treat
      individual object trajectories as a training example rather than an entire
      scenario.
  """

  path: str
  data_format: DataFormat = DataFormat.TFRECORD
  repeat: Optional[int] = None
  batch_dims: tuple[int, ...] = ()
  shuffle_seed: Optional[int] = None
  shuffle_buffer_size: int = 1_000
  num_shards: int = 4
  deterministic: bool = True
  include_sdc_paths: bool = False
  aggregate_timesteps: bool = True
  max_num_rg_points: int = 30000
  max_num_objects: Optional[int] = None
  num_paths: Optional[int] = None
  num_points_per_path: Optional[int] = None
  drop_remainder: bool = True
  tf_data_service_address: Optional[str] = None
  distributed: bool = False
  batch_by_scenario: bool = True


@dataclasses.dataclass(frozen=True)
class ObservationConfig:
  """Config for the built-in Waymax Observation function.

  Attributes:
    num_steps: Number of trajectory history steps for observation.
    roadgraph_top_k: Number of closest roadgraph elements to get for
      observation.
    coordinate_frame: What coordinate frame the observation is using.
  """

  num_steps: int = 1
  roadgraph_top_k: int = 2000
  coordinate_frame: CoordinateFrame = CoordinateFrame.GLOBAL


@dataclasses.dataclass(frozen=True)
class MetricsConfig:
  """Config for the built-in Waymax Metrics functions.

  Attributes:
    run_log_divergence: Whether log_divergence metric will be computed in the
      `step` function.
    run_overlap: Whether overlap metric will be computed in the `step` function.
    run_offroad: Whether offroad metric will be computed in the `step` function.
    run_sdc_wrongway: Whether wrong-way metric will be computed for SDC in the
      `step` function. Note this is only for single-agent env currently since
      there is no route for sim-agents in data.
    run_sdc_progression: Whether progression metric will be computed for SDC in
      the `step` function. Note this is only for single-agent env currently
      since there is no route for sim-agents in data.
    run_sdc_off_route: Whether the off-route metric will be computed for SDC in
      the `step` function. Note this is only for single-agent env currently
      since there is no route for sim-agents in data.
    run_sdc_kinematic_infeasibility: Whether the kinematics infeasibility metric
      will be computed for SDC in the `step` function. Note this is only for
      single-agent env currently since other agents may have different dynamics
      and cannot be evaluated using the current kinematics infeasibility metrics
  """

  run_log_divergence: bool = True
  run_overlap: bool = True
  run_offroad: bool = True
  run_sdc_wrongway: bool = False
  run_sdc_progression: bool = False
  run_sdc_off_route: bool = False
  run_sdc_kinematic_infeasibility: bool = False


@dataclasses.dataclass(frozen=True)
class LinearCombinationRewardConfig:
  """Config listing all metrics and their corresponding weights.

  Attributes:
    rewards: Dictionary of metric names to floats indicating the weight of each
      metric to create a reward of a linear combination. Valid metric names are
      taken from the MetricConfig and removing 'run_'. For example, to create a
      reward using the progression metric, the name would have to be
      'sdc_progression', since 'run_sdc_progression' is used in the config
      above.
  """

  rewards: dict[str, float]


class ObjectType(enum.Enum):
  """Types of objects that can be controlled by Waymax."""

  SDC = 'SDC'  # The ego-vehicle.
  MODELED = 'MODELED'  # tracks_to_predict objects defined in WOMD.
  VALID = 'VALID'  # All valid objects in the scene.
  NON_SDC = 'NON_SDC'  # Any objects expect the SDC (ego-vehicle).


class SimAgentType(enum.Enum):
  """Types of sim agents that can be used by Waymax."""

  IDM = 'IDM'  # Intelligent driver model.


@dataclasses.dataclass(frozen=True)
class SimAgentConfig:
  """Config for sim agents.

  Attributes:
    agent_type: What sim agent algorithm to use.
    controlled_objects: Which objects the sim agent should control.
  """

  agent_type: SimAgentType
  controlled_objects: ObjectType


@dataclasses.dataclass(frozen=True)
class EnvironmentConfig:
  """Configs for Waymax Environments.

  Attributes:
    max_num_objects: Max number of objects in the scene, should be consistent
      with DatasetConfig.max_num_objects.
    init_steps: Initial/warmup steps taken for the environment. For WOMD, this
      includes 10 warmup steps and 1 for the current step
    controlled_object: What type of objects are controlled.
    compute_reward: Whether to compute the reward. Computing the reward is
      expensive and unnecessary for cases like rollout data generation.
    allow_new_objects_after_warmup: Whether to allow new objects to appear after
      environment warmup. If this is set to `False`, all non-controlled objects
      that are invalid in the log at timestep `t=warmup` will be forever invalid
      in the simulation. This means that objects that appear out of occlusion
      after `t=warmup` will still be invalid as if they never appeared. If this
      is set to `True`, agents will be able to appear in the future simulated
      state if they appeared in the future logged state. Note when set to True,
      the environment could expect users to control objects even before their
      first appearance, users can either ignore or simply provide any invalid
      action for those objects.
    observation: Optional config for the provided observation function found at
      waymax/data/observation.py.
    metrics: Specifies which built-in Waymax metrics to run when calling
      `env.metrics(...)`.
    rewards: Specifies the metrics and weights to create a reward as a linear
      combination of metrics.
    sim_agents: Configurations for sim agents used to control non
      user-controlled objects. Sim agents are applied in the order of that they
      are specified (if multiple sim agents control the same object, only the
      last sim agent will be applied for that object).
  """

  max_num_objects: int = 128
  init_steps: int = 11
  controlled_object: ObjectType = ObjectType.SDC
  compute_reward: bool = True
  allow_new_objects_after_warmup: bool = True
  observation: Optional[ObservationConfig] = None
  metrics: MetricsConfig = MetricsConfig()
  rewards: LinearCombinationRewardConfig = LinearCombinationRewardConfig(
      rewards={'overlap': -1.0, 'offroad': -1.0}
  )
  sim_agents: Optional[Sequence[SimAgentConfig]] = None

  def __post_init__(self):
    if self.observation is not None:
      if self.observation.num_steps > self.init_steps:
        raise ValueError(
            'Initial steps must be greater than the number of '
            'history steps. Please set init_steps >= obs_num_steps.'
        )


@dataclasses.dataclass(frozen=True)
class WaymaxConfig:
  """Top level config for Waymax.

  Attributes:
    data_config: Data related configurations, including how to parse and load
      the data.
    env_config: Configurations about the environment itself, observation, and
      metrics.
  """

  data_config: DatasetConfig
  env_config: EnvironmentConfig

  def __post_init__(self):
    if not self.data_config.include_sdc_paths and (
        self.env_config.metrics.run_sdc_wrongway
        | self.env_config.metrics.run_sdc_progression
        | self.env_config.metrics.run_sdc_off_route
    ):
      raise ValueError(
          'Need to set data_config.include_sdc_paths True in  '
          'order to compute route based metrics for SDC.'
      )

WOD_1_0_0_TRAINING = DatasetConfig(
    path='gs://waymo_open_dataset_motion_v_1_0_0/uncompressed/tf_example/training/training_tfexample.tfrecord@1000',
    max_num_rg_points=20000,
    data_format=DataFormat.TFRECORD,
)

WOD_1_0_0_VALIDATION = DatasetConfig(
    path='gs://waymo_open_dataset_motion_v_1_0_0/uncompressed/tf_example/validation/validation_tfexample.tfrecord@150',
    max_num_rg_points=20000,
    data_format=DataFormat.TFRECORD,
)

WOD_1_1_0_TRAINING = DatasetConfig(
    path='gs://waymo_open_dataset_motion_v_1_1_0/uncompressed/tf_example/training/training_tfexample.tfrecord@1000',
    max_num_rg_points=20000,
    data_format=DataFormat.TFRECORD,
)

WOD_1_1_0_VALIDATION = DatasetConfig(
    path='gs://waymo_open_dataset_motion_v_1_1_0/uncompressed/tf_example/validation/validation_tfexample.tfrecord@150',
    max_num_rg_points=20000,
    data_format=DataFormat.TFRECORD,
)

WOD_1_2_0_TEST = DatasetConfig(
    path='gs://waymo_open_dataset_motion_v_1_2_0/uncompressed/tf_example/testing/testing_tfexample.tfrecord@150',
    max_num_rg_points=30000,
    data_format=DataFormat.TFRECORD,
)
