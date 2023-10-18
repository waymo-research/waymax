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

"""Util functions for general dataloading."""

import functools
import math
import os
import random
from typing import Callable, Iterator, Optional, Sequence, TypeVar

import jax
import tensorflow as tf

from waymax import config as _config


T = TypeVar('T')
AUTOTUNE = tf.data.AUTOTUNE


def generate_sharded_filenames(path: str) -> Sequence[str]:
  """Returns the filenames of individual sharded files.

  A sharded file is a set of files of the format filename-XXXXX-of-YYYYY,
  where XXXXX is a placeholder for the index of the shard, and YYYYY is the
  total number of shards. These files are collectively referred to by a
  sharded path filename@YYYYY.

  For example, the sharded path `myfile@100` refers to the set of files
    - myfile-00000-of-00100
    - myfile-00001-of-00100
    - ...
    - myfile-00098-of-00100
    - myfile-00099-of-00100

  Args:
    path: A path to a sharded file, with format `filename@shards`, where shards
      is an integer denoting the number of total shards.

  Returns:
    An iterator through the complete set of filenames that the path refers to,
    with each filename having the format `filename-XXXXX-of-YYYYY`
  """
  base_name, num_shards = path.split('@')
  num_shards = int(num_shards)
  shard_width = max(5, int(math.log10(num_shards) + 1))
  format_str = base_name + '-%0' + str(shard_width) + 'd-of-%05d'
  return [format_str % (i, num_shards) for i in range(num_shards)]


def tf_examples_dataset(
    path: str,
    data_format: _config.DataFormat,
    preprocess_fn: Callable[[bytes], dict[str, tf.Tensor]],
    shuffle_seed: Optional[int] = None,
    shuffle_buffer_size: int = 100,
    repeat: Optional[int] = None,
    batch_dims: Sequence[int] = (),
    num_shards: int = 1,
    deterministic: bool = True,
    drop_remainder: bool = True,
    tf_data_service_address: Optional[str] = None,
    batch_by_scenario: bool = True,
) -> tf.data.Dataset:
  """Returns a dataset of Open Motion dataset TFExamples.

  Each TFExample contains data for the trajectory of all objects, the roadgraph,
  and traffic light states. See https://waymo.com/open/data/motion/tfexample
  for the data format definition.

  Args:
    path: The path to the dataset.
    data_format: Data format of the dataset.
    preprocess_fn: Function for parsing and preprocessing individual examples.
    shuffle_seed: Seed for shuffling. If left default (None), will not shuffle
      the dataset.
    shuffle_buffer_size: The size of the shuffle buffer.
    repeat: Number of times to repeat the dataset. Default (None) will repeat
      infinitely.
    batch_dims: List of size of batch dimensions. Multiple batch dimension can
      be used to provide inputs for multiple devices. E.g.
      [jax.local_device_count(), batch_size_per_device].
    num_shards: Number of shards for parallel loading, no effect on data
      returned.
    deterministic: Whether to use deterministic parallel processing.
    drop_remainder: Arg for tf.data.Dataset.batch. Set True to drop remainder if
      the last batch does not contains enough examples.
    tf_data_service_address: Set to use tf data service.
    batch_by_scenario: If True, one example in a returned batch is the entire
      scenario containing all objects; if False, the dataset will treat
      individual object trajectories as a training example rather than an entire
      scenario.

  Returns:
    A tf.data.Dataset of Open Motion Dataset tf.Example elements.
  """

  if data_format == _config.DataFormat.TFRECORD:
    dataset_fn = tf.data.TFRecordDataset
  else:
    raise ValueError('Data format %s is not supported.' % data_format)

  files_to_load = [path]
  if '@' in os.path.basename(path):
    files_to_load = generate_sharded_filenames(path)
  if shuffle_seed:
    random.seed(shuffle_seed)
    random.shuffle(files_to_load)
  files = tf.data.Dataset.from_tensor_slices(files_to_load)
  # Split files across multiple processes for distributed training/eval.
  files = files.shard(jax.process_count(), jax.process_index())

  def _make_dataset(
      shard_index: int, num_shards: int, local_files: tf.data.Dataset
  ):
    # Outer parallelism.
    local_files = local_files.shard(num_shards, shard_index)
    ds = local_files.interleave(
        dataset_fn,
        num_parallel_calls=AUTOTUNE,
        cycle_length=AUTOTUNE,
        deterministic=deterministic,
    )

    ds = ds.repeat(repeat)
    if shuffle_seed is not None:
      # Makes sure each host uses a different RNG for shuffling.
      local_seed = jax.random.fold_in(
          jax.random.PRNGKey(shuffle_seed), jax.process_index()
      )[0]
      ds = ds.shuffle(shuffle_buffer_size, seed=local_seed)

    ds = ds.map(
        preprocess_fn, num_parallel_calls=AUTOTUNE, deterministic=deterministic
    )
    if not batch_by_scenario:
      ds = ds.unbatch()
    if batch_dims:
      for batch_size in reversed(batch_dims):
        ds = ds.batch(
            batch_size,
            drop_remainder=drop_remainder,
            num_parallel_calls=AUTOTUNE,
            deterministic=deterministic,
        )
    return ds

  make_dataset_fn = functools.partial(
      _make_dataset, num_shards=num_shards, local_files=files
  )
  indices = tf.data.Dataset.range(num_shards)
  dataset = indices.interleave(
      make_dataset_fn, num_parallel_calls=AUTOTUNE, deterministic=deterministic
  )

  if tf_data_service_address is not None:
    dataset = dataset.apply(
        tf.data.experimental.service.distribute(
            processing_mode=tf.data.experimental.service.ShardingPolicy.OFF,
            service=tf_data_service_address,
        )
    )
  return dataset.prefetch(AUTOTUNE)


def get_data_generator(
    config: _config.DatasetConfig,
    preprocess_fn: Optional[
        Callable[[bytes], dict[str, tf.Tensor | dict[str, tf.Tensor]]]
    ],
    postprocess_fn: Optional[Callable[[dict[str, jax.Array]], T]] = None,
) -> Iterator[T]:
  """Iterator that yields the desired object returned by postprocess_fn.

  It parses data using preprocess_fn and returns a generator of data whose data
  structure is defined by postprocess_fn function.

  Args:
    config: config for dataset and preprocessing.
    preprocess_fn: preprocess the serialized data into a dictionary of str to tf
      Tensor.
    postprocess_fn: a function that converts dict of jnp array to desired data
      class. Note for distributed training, this function will be pmap-ed and
      executed in the main process.

  Yields:
    Iterator of desired data class.
  """

  batch_dims = config.batch_dims
  if config.distributed:
    batch_dims = (jax.local_device_count(),) + batch_dims

  dataset = tf_examples_dataset(
      path=config.path,
      data_format=config.data_format,
      preprocess_fn=preprocess_fn,
      shuffle_seed=config.shuffle_seed,
      shuffle_buffer_size=config.shuffle_buffer_size,
      repeat=config.repeat,
      batch_dims=batch_dims,
      num_shards=config.num_shards,
      deterministic=config.deterministic,
      drop_remainder=config.drop_remainder,
      tf_data_service_address=config.tf_data_service_address,
      batch_by_scenario=config.batch_by_scenario,
  )

  if postprocess_fn is None:
    for example in dataset.as_numpy_iterator():
      yield example
  else:
    if config.distributed:
      postprocess_fn = jax.pmap(postprocess_fn, axis_name='batch')
    else:
      postprocess_fn = jax.jit(postprocess_fn)
    for example in dataset.as_numpy_iterator():
      yield postprocess_fn(example)
