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

from absl.testing import absltest
from absl.testing import parameterized
from waymax import config as waymax_config
from waymax import dataloader
from waymax import datatypes
from waymax.utils import test_utils
from waymax.visualization import viz


class VizTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    config = waymax_config.DatasetConfig(
        path=test_utils.ROUTE_DATA_PATH,
        data_format=waymax_config.DataFormat.TFRECORD,
    )
    state_t0 = next(dataloader.simulator_state_generator(config))
    self.state_t10 = datatypes.update_state_by_log(state_t0, 10)

  def test_plot_sim_state_generates_np_img(self):
    img = viz.plot_simulator_state(self.state_t10)
    self.assertTupleEqual(img.shape, (600, 600, 3))

  @parameterized.parameters(
      waymax_config.CoordinateFrame.SDC,
      waymax_config.CoordinateFrame.OBJECT,
      waymax_config.CoordinateFrame.GLOBAL,
  )
  def test_plot_observation_generates_np_img(self, coordinate_frame):
    obs = datatypes.observation_from_state(
        self.state_t10,
        obs_num_steps=5,
        roadgraph_top_k=100,
        coordinate_frame=coordinate_frame,
    )
    img = viz.plot_observation(obs, obj_idx=0)
    self.assertTupleEqual(img.shape, (600, 600, 3))

  @parameterized.parameters(
      waymax_config.CoordinateFrame.SDC,
      waymax_config.CoordinateFrame.OBJECT,
      waymax_config.CoordinateFrame.GLOBAL,
  )
  def test_plot_sdc_observation_generates_np_img(self, coordinate_frame):
    obs = datatypes.sdc_observation_from_state(
        self.state_t10,
        obs_num_steps=10,
        roadgraph_top_k=100,
        coordinate_frame=coordinate_frame,
    )
    img = viz.plot_observation(obs, obj_idx=0)
    self.assertTupleEqual(img.shape, (600, 600, 3))


if __name__ == '__main__':
  absltest.main()
