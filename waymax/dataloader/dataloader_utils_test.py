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

import functools

import tensorflow as tf

from absl.testing import parameterized
from waymax import config as _config
from waymax.dataloader import dataloader_utils
from waymax.dataloader import womd_dataloader
from waymax.utils import test_utils


class DataloaderUtilsTest(parameterized.TestCase, tf.test.TestCase):

  def test_dataloader_without_batch_dim(self):
    config = _config.DatasetConfig(
        path=test_utils.ROUTE_DATA_PATH,
        data_format=_config.DataFormat.TFRECORD,
        aggregate_timesteps=False,
    )
    preprocess_fn = functools.partial(
        womd_dataloader.preprocess_serialized_womd_data, config=config
    )
    dataset = dataloader_utils.tf_examples_dataset(
        path=test_utils.ROUTE_DATA_PATH,
        data_format=_config.DataFormat.TFRECORD,
        preprocess_fn=preprocess_fn,
    )
    # Check consistency with WOMD original data format.
    self.assertEqual(dataset.element_spec['state/past/x'].shape, (128, 10))
    self.assertEqual(dataset.element_spec['state/current/x'].shape, (128, 1))
    self.assertEqual(dataset.element_spec['state/future/x'].shape, (128, 80))
    self.assertEqual(
        dataset.element_spec['traffic_light_state/past/x'].shape, (10, 16)
    )
    self.assertEqual(
        dataset.element_spec['roadgraph_samples/id'].shape,
        (30000, 1),
    )
    # Deterministic check on a sample value
    example = dataset.take(1).get_single_element()
    self.assertTupleEqual(
        tuple(example['state/past/x'][0, :3]), (1120.4044, 1119.4, 1118.4131)
    )

  def test_dataloader_with_two_batch_dim(self):
    config = _config.DatasetConfig(
        path=test_utils.ROUTE_DATA_PATH,
        data_format=_config.DataFormat.TFRECORD,
        aggregate_timesteps=False,
    )
    preprocess_fn = functools.partial(
        womd_dataloader.preprocess_serialized_womd_data, config=config
    )
    dataset = dataloader_utils.tf_examples_dataset(
        path=test_utils.ROUTE_DATA_PATH,
        data_format=_config.DataFormat.TFRECORD,
        preprocess_fn=preprocess_fn,
        batch_dims=(2, 1),
    )
    self.assertEqual(
        dataset.element_spec['state/past/x'].shape, (2, 1, 128, 10)
    )
    self.assertEqual(
        dataset.element_spec['state/current/x'].shape, (2, 1, 128, 1)
    )
    self.assertEqual(
        dataset.element_spec['state/future/x'].shape, (2, 1, 128, 80)
    )
    self.assertEqual(
        dataset.element_spec['traffic_light_state/past/x'].shape, (2, 1, 10, 16)
    )
    self.assertEqual(
        dataset.element_spec['roadgraph_samples/id'].shape,
        (2, 1, 30000, 1),
    )
    # Deterministic check on a sample value
    example = dataset.take(1).get_single_element()
    self.assertTupleEqual(
        tuple(example['state/past/x'][0, 0, 0, :3]),
        (1120.4044, 1119.4, 1118.4131),
    )

  def test_dataloader_with_time_aggregated(self):
    config = _config.DatasetConfig(
        path=test_utils.ROUTE_DATA_PATH,
        data_format=_config.DataFormat.TFRECORD,
        aggregate_timesteps=True,
    )
    preprocess_fn = functools.partial(
        womd_dataloader.preprocess_serialized_womd_data, config=config
    )
    dataset = dataloader_utils.tf_examples_dataset(
        path=test_utils.ROUTE_DATA_PATH,
        data_format=_config.DataFormat.TFRECORD,
        preprocess_fn=preprocess_fn,
    )
    self.assertEqual(dataset.element_spec['state/all/x'].shape, (128, 91))
    self.assertEqual(
        dataset.element_spec['traffic_light_state/all/x'].shape, (91, 16)
    )
    self.assertEqual(
        dataset.element_spec['roadgraph_samples/id'].shape,
        (30000, 1),
    )
    # Deterministic check on a sample value
    example = dataset.take(1).get_single_element()
    self.assertTupleEqual(
        tuple(example['state/all/x'][0, :3]), (1120.4044, 1119.4, 1118.4131)
    )


if __name__ == '__main__':
  tf.test.main()
