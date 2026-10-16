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

"""Regression tests for the WOSAC notebook's scenario-ID dataloader."""

import itertools
import json
import math
from pathlib import Path
import tempfile
from unittest import mock

from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf

from waymax import config as _config
from waymax import dataloader
from waymax.utils import test_utils


_SCENARIO_IDS = (
    'a',
    '123456789abcdef',
    '123456789abcdef0',
    '123456789abcdef0123456789',
    '000000000000000f',
    '',
    'scénario-道路',
    'nul\x00tail\x00',
    'another-id',
    'final-id',
)
_BATCH_CASES = [
    dict(
        testcase_name=f'{name}_{"drop" if drop else "keep"}_remainder',
        batch_dims=dims,
        drop_remainder=drop,
    )
    for name, dims in (
        ('unbatched', ()),
        ('one', (1,)),
        ('two', (2,)),
        ('three', (3,)),
        ('nested', (2, 2)),
    )
    for drop in (True, False)
]


class WosacNotebookDataloaderTest(parameterized.TestCase, tf.test.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    notebook_path = Path(__file__).with_name(
        'wosac_submission_via_waymax.ipynb'
    )
    notebook = json.loads(notebook_path.read_text(encoding='utf-8'))
    cls.loader_source = next(
        ''.join(cell['source'])
        for cell in notebook['cells']
        if cell.get('metadata', {}).get('id') == 'JJAfGGSF74Ym'
    )
    cls.serialized_example = next(
        iter(tf.data.TFRecordDataset(test_utils.ROUTE_DATA_PATH))
    ).numpy()

  def _make_config(
      self,
      batch_dims=(),
      drop_remainder=False,
      distributed=False,
      scenario_ids=None,
  ):
    # Ten examples permit a partial outer nested batch with full inner batches.
    # Other cases use seven examples to exercise ordinary partial batches.
    count = 10 if len(batch_dims) > 1 else 7
    if distributed:
      count = max(count, 2 * jax.local_device_count() + 1)
    if scenario_ids is None:
      scenario_ids = tuple(
          _SCENARIO_IDS[index % len(_SCENARIO_IDS)] for index in range(count)
      )
    count = len(scenario_ids)
    directory = tempfile.TemporaryDirectory()
    self.addCleanup(directory.cleanup)
    path = str(Path(directory.name) / 'scenarios.tfrecord')
    with tf.io.TFRecordWriter(path) as writer:
      for index, scenario_id in enumerate(scenario_ids):
        example = tf.train.Example.FromString(self.serialized_example)
        example.features.feature['scenario/id'].bytes_list.value[:] = [
            scenario_id.encode('utf-8')
        ]
        # Make ID-to-state alignment observable even when source data repeats.
        example.features.feature['state/current/x'].float_list.value[0] = index
        writer.write(example.SerializeToString())
    return _config.DatasetConfig(
        path=path,
        repeat=1,
        num_shards=1,
        max_num_objects=2,
        batch_dims=batch_dims,
        drop_remainder=drop_remainder,
        distributed=distributed,
    ), count

  def _load_notebook(self, config):
    namespace = dict(
        tf=tf, jax=jax, jnp=jnp, np=np, _config=_config, dataloader=dataloader
    )
    code = compile(self.loader_source, 'wosac_dataloader_cell', 'exec')
    with mock.patch.object(_config, 'WOD_1_2_0_TESTING', config):
      exec(code, namespace)
    return namespace

  @parameterized.named_parameters(*_BATCH_CASES)
  def test_ids_and_states_preserve_batching(self, batch_dims, drop_remainder):
    config, count = self._make_config(batch_dims, drop_remainder)
    notebook = self._load_notebook(config)
    baseline = dataloader.simulator_state_generator(config)
    seen_ids = []
    missing = object()
    for result, expected_state in itertools.zip_longest(
        notebook['data_iter'], baseline, fillvalue=missing
    ):
      self.assertIsNot(result, missing)
      self.assertIsNot(expected_state, missing)
      scenario_ids, state = result
      self.assertEqual(np.asarray(scenario_ids).shape, state.shape)
      if not batch_dims:
        self.assertIsInstance(scenario_ids, str)
      current_ids = np.asarray(scenario_ids).reshape(-1).tolist()
      start = len(seen_ids)
      self.assertEqual(
          current_ids, list(_SCENARIO_IDS[start:start + len(current_ids)])
      )
      self.assertAllEqual(
          np.asarray(state.log_trajectory.x[..., 0, 10]).reshape(-1),
          np.arange(start, start + len(current_ids)),
      )
      jax.tree_util.tree_map(self.assertAllEqual, state, expected_state)
      seen_ids.extend(current_ids)
    expected_count = count
    if batch_dims and drop_remainder:
      expected_count = count // math.prod(batch_dims) * math.prod(batch_dims)
    self.assertEqual(seen_ids, list(_SCENARIO_IDS[:expected_count]))

  def test_preprocessor_preserves_string_ids(self):
    config, count = self._make_config()
    notebook = self._load_notebook(config)
    serialized = tf.data.TFRecordDataset(config.path)
    for record, expected_id in zip(serialized, _SCENARIO_IDS[:count]):
      scenario_id = notebook['_preprocess'](record)['scenario/id']
      self.assertEqual(scenario_id.dtype, tf.string)
      self.assertEqual(scenario_id.shape, (1,))
      self.assertEqual(scenario_id.numpy()[0].decode('utf-8'), expected_id)

  def test_equal_length_ids_remain_separate(self):
    expected_ids = ('123456789abcdef', 'fedcba987654321')
    config, _ = self._make_config(
        batch_dims=(2,), scenario_ids=expected_ids
    )
    notebook = self._load_notebook(config)
    scenario_ids, state = next(notebook['data_iter'])
    self.assertEqual(state.shape, (2,))
    self.assertEqual(np.asarray(scenario_ids).shape, state.shape)
    self.assertAllEqual(scenario_ids, expected_ids)
    self.assertAllEqual(state.log_trajectory.x[..., 0, 10], (0, 1))
    self.assertEmpty(list(notebook['data_iter']))

  def test_unbatched_id_preserves_embedded_and_trailing_nuls(self):
    expected_id = 'nul\x00tail\x00'
    config, _ = self._make_config(scenario_ids=(expected_id,))
    notebook = self._load_notebook(config)
    scenario_id, state = next(notebook['data_iter'])
    self.assertIsInstance(scenario_id, str)
    self.assertEqual(scenario_id, expected_id)
    self.assertEqual(state.shape, ())
    self.assertEqual(state.log_trajectory.x[0, 10], 0)
    self.assertEmpty(list(notebook['data_iter']))

  def test_default_loader_batches_mixed_length_ids(self):
    config, count = self._make_config(batch_dims=(3,))
    states = list(dataloader.simulator_state_generator(config))
    self.assertEqual([state.shape for state in states], [(3,), (3,), (1,)])
    self.assertAllEqual(
        np.concatenate([
            np.asarray(state.log_trajectory.x[..., 0, 10]) for state in states
        ]),
        np.arange(count),
    )

  def test_distributed_ids_match_device_and_batch_axes(self):
    config, count = self._make_config(
        batch_dims=(2,), drop_remainder=True, distributed=True
    )
    notebook = self._load_notebook(config)
    batch_size = jax.local_device_count() * 2
    expected_count = count // batch_size * batch_size
    self.assertGreater(expected_count, 0)
    expected_ids = [
        _SCENARIO_IDS[index % len(_SCENARIO_IDS)]
        for index in range(expected_count)
    ]
    seen_ids = []
    for scenario_ids, state in notebook['data_iter']:
      self.assertEqual(state.shape, (jax.local_device_count(), 2))
      self.assertEqual(np.asarray(scenario_ids).shape, state.shape)
      current_ids = np.asarray(scenario_ids).reshape(-1).tolist()
      start = len(seen_ids)
      self.assertEqual(
          current_ids, expected_ids[start:start + batch_size]
      )
      self.assertAllEqual(
          np.asarray(state.log_trajectory.x[..., 0, 10]).reshape(-1),
          np.arange(start, start + batch_size),
      )
      seen_ids.extend(current_ids)
    self.assertEqual(seen_ids, expected_ids)


if __name__ == '__main__':
  tf.test.main()
