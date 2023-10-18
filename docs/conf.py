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

"""Sphinx configuration file."""

import os
import sys

# Source code dir relative to this file
sys.path.insert(0, os.path.abspath('..'))

project = 'Waymax'
copyright = '2023, The Waymax Authors'  # pylint: disable=redefined-builtin
author = 'The Waymax Authors'
version = ''
release = ''
language = 'en'


extensions = [
    'autoapi.extension',
    'sphinx.ext.napoleon',
    'sphinx_design',  # Enables grids.
    'myst_nb',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# AutoAPI options
autoapi_type = 'python'
autoapi_dirs = ['../waymax']
autoapi_ignore = ['*_test.py']
autoapi_add_toctree_entry = False
autoapi_keep_files = True

# Theme Options
html_theme = 'furo'
html_theme_options = {
    'source_repository': 'https://github.com/waymo-research/waymax'
}
html_favicon = 'https://waymo.com/favicon.png'

# -- Options for myst ----------------------------------------------
myst_heading_anchors = 3  # auto-generate 3 levels of heading anchors
myst_enable_extensions = ['dollarmath']
nb_execution_mode = 'off'
nb_execution_timeout = 300
nb_execution_allow_errors = False
nb_merge_streams = False

# The suffix(es) of source filenames.
# Note: important to list ipynb before md here: we have both md and ipynb
# copies of each notebook, and myst will choose which to convert based on
# the order in the source_suffix list. Notebooks which are not executed have
# outputs stored in ipynb but not in md, so we must convert the ipynb.
source_suffix = ['.rst', '.ipynb', '.md']
