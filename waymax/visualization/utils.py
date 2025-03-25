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

"""General visualization functions for non-waymax data using matplotlib.

Note there is no batch dimension, and should not rely on any customized data
structure.
"""

import colorsys
import dataclasses
import os
from typing import Optional

import matplotlib
import matplotlib.pylab as plt
import numpy as np
from PIL import Image


@dataclasses.dataclass
class VizConfig:
  """Config for visualization."""

  front_x: float = 75.0
  back_x: float = 75.0
  front_y: float = 75.0
  back_y: float = 75.0
  px_per_meter: float = 4.0
  show_agent_id: bool = True
  center_agent_idx: int = -1  # -1 for SDC
  verbose: bool = True


def init_fig_ax_via_size(
    x_px: float, y_px: float
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
  """Initializes a figure with given size in pixel."""
  fig, ax = plt.subplots()
  # Sets output image to pixel resolution.
  dpi = 100
  fig.set_size_inches([x_px / dpi, y_px / dpi])
  fig.set_dpi(dpi)
  fig.set_facecolor('white')
  return fig, ax


def init_fig_ax(
    vis_config: VizConfig = VizConfig(),
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
  """Initializes a figure with vis_config."""
  return init_fig_ax_via_size(
      (vis_config.front_x + vis_config.back_x) * vis_config.px_per_meter,
      (vis_config.front_y + vis_config.back_y) * vis_config.px_per_meter,
  )


def center_at_xy(
    ax: matplotlib.axes.Axes,
    xy: np.ndarray,
    vis_config: VizConfig = VizConfig(),
) -> None:
  ax.axis((
      xy[0] - vis_config.back_x,
      xy[0] + vis_config.front_x,
      xy[1] - vis_config.back_y,
      xy[1] + vis_config.front_y,
  ))


def img_from_fig(fig: matplotlib.figure.Figure) -> np.ndarray:
  """Returns a [H, W, 3] uint8 np image from fig.canvas.tostring_argb()."""
  # Just enough margin in the figure to display xticks and yticks.
  fig.subplots_adjust(
      left=0.08, bottom=0.08, right=0.98, top=0.98, wspace=0.0, hspace=0.0
  )
  fig.canvas.draw()
  data = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
  img = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, 1:]
  plt.close(fig)
  return img


def save_img_as_png(img: np.ndarray, filename: str = '/tmp/img.png'):
  """Saves np image to disk."""
  outdir = os.path.dirname(filename)
  os.makedirs(outdir, exist_ok=True)
  Image.fromarray(img).save(filename)


def get_n_colors(
    num_color: int = 10,
    max_hue: float = 1.0,
    saturation: float = 1.0,
    lightness: float = 1.0,
) -> np.ndarray:
  """Gets n different colors."""
  hsv_list = [
      (x * max_hue / num_color, saturation, lightness) for x in range(num_color)
  ]
  return np.array([colorsys.hsv_to_rgb(*x) for x in hsv_list])


def plot_numpy_bounding_boxes(
    ax: matplotlib.axes.Axes,
    bboxes: np.ndarray,
    color: np.ndarray,
    alpha: Optional[float] = 1.0,
    as_center_pts: bool = False,
    label: Optional[str] = None,
) -> None:
  """Plots multiple bounding boxes.

  Args:
    ax: Fig handles.
    bboxes: Shape (num_bbox, 5), with last dimension as (x, y, length, width,
      yaw).
    color: Shape (3,), represents RGB color for drawing.
    alpha: Alpha value for drawing, i.e. 0 means fully transparent.
    as_center_pts: If set to True, bboxes will be drawn as center points,
      instead of full bboxes.
    label: String, represents the meaning of the color for different boxes.
  """
  if bboxes.ndim != 2 or bboxes.shape[1] != 5 or color.shape != (3,):
    raise ValueError(
        (
            'Expect bboxes rank 2, last dimension of bbox 5, color of size 3,'
            ' got{}, {}, {} respectively'
        ).format(bboxes.ndim, bboxes.shape[1], color.shape)
    )

  if as_center_pts:
    ax.plot(
        bboxes[:, 0],
        bboxes[:, 1],
        'o',
        color=color,
        ms=2,
        alpha=alpha,
        label=label,
    )
  else:
    c = np.cos(bboxes[:, 4])
    s = np.sin(bboxes[:, 4])
    pt = np.array((bboxes[:, 0], bboxes[:, 1]))  # (2, N)
    length, width = bboxes[:, 2], bboxes[:, 3]
    u = np.array((c, s))
    ut = np.array((s, -c))

    # Compute box corner coordinates.
    tl = pt + length / 2 * u - width / 2 * ut
    tr = pt + length / 2 * u + width / 2 * ut
    br = pt - length / 2 * u + width / 2 * ut
    bl = pt - length / 2 * u - width / 2 * ut

    # Compute heading arrow using center left/right/front.
    cl = pt - width / 2 * ut
    cr = pt + width / 2 * ut
    cf = pt + length / 2 * u

    # Draw bboxes.
    ax.plot(
        [tl[0, :], tr[0, :], br[0, :], bl[0, :], tl[0, :]],
        [tl[1, :], tr[1, :], br[1, :], bl[1, :], tl[1, :]],
        color=color,
        zorder=4,
        alpha=alpha,
        label=label,
    )

    # Draw heading arrow.
    ax.plot(
        [cl[0, :], cr[0, :], cf[0, :], cl[0, :]],
        [cl[1, :], cr[1, :], cf[1, :], cl[1, :]],
        color=color,
        zorder=4,
        alpha=alpha,
        label=label,
    )
