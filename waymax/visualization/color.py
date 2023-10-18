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

"""Defines the color for different components."""


from immutabledict import immutabledict
import numpy as np


TRAFFIC_LIGHT_COLORS = {
    # Unknown = 0, Arrow_Stop = 1, Arrow_Caution = 2, Arrow_Go = 3, Stop = 4,
    # Caution = 5, Go = 6, Flashing_Stop = 7, Flashing_Caution = 8
    # third_party/waymo_open_dataset/protos/map.proto
    0: [0.75, 0.75, 0.75],
    1: [1.0, 0.0, 0.0],
    2: [1.0, 1.0, 0.0],
    3: [0.0, 1.0, 0.0],
    4: [1.0, 0.0, 0.0],
    5: [1.0, 1.0, 0.0],
    6: [0.0, 1.0, 0.0],
    7: [1.0, 1.0, 0.0],
    8: [1.0, 1.0, 0.0],
}

ROAD_GRAPH_COLORS = {
    # Consistent with MapElementIds
    1: np.array([230, 230, 230]) / 255.0,  # 'LaneCenter-Freeway',
    2: np.array([230, 230, 230]) / 255.0,  # 'LaneCenter-SurfaceStreet',
    3: np.array([230, 230, 230]) / 255.0,  # 'LaneCenter-BikeLane',
    6: np.array([140, 230, 255]) / 255.0,  # 'RoadLine-BrokenSingleWhite',
    7: np.array([89, 219, 255]) / 255.0,  # 'RoadLine-SolidSingleWhite',
    8: np.array([89, 219, 255]) / 255.0,  # 'RoadLine-SolidDoubleWhite',
    9: np.array([241, 153, 255]) / 255.0,  # 'RoadLine-BrokenSingleYellow',
    10: np.array([241, 153, 255]) / 255.0,  # 'RoadLine-BrokenDoubleYellow'
    11: np.array([120, 120, 120]) / 255.0,  # 'RoadLine-SolidSingleYellow',
    12: np.array([120, 120, 120]) / 255.0,  # 'RoadLine-SolidDoubleYellow',
    13: np.array([120, 120, 120]) / 255.0,  # 'RoadLine-PassingDoubleYellow',
    15: np.array([80, 80, 80]) / 255.0,  # 'RoadEdgeBoundary',
    16: np.array([80, 80, 80]) / 255.0,  # 'RoadEdgeMedian',
    17: np.array([255, 0, 0]) / 255.0,  # 'StopSign',  # One point
    18: np.array([200, 200, 200]) / 255.0,  # 'Crosswalk',  # Polygon
    19: np.array([200, 200, 200]) / 255.0,  # 'SpeedBump',  # Polygon
}


COLOR_DICT = immutabledict({
    # RGB color:
    'context': np.array([0.6, 0.6, 0.6]),  # Context agents, grey.
    'controlled': np.array([0, 0.6, 0.8]),  # Modeled agents, dark blue.
    'history': np.array([0.8, 0.8, 0.8]),  # Grey for history.
    'overlap': np.array([1.0, 0.0, 0.0]),  # Red for overlap
})
