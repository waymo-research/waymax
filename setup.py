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

"""setup.py file for Waymax."""
from setuptools import find_packages
from setuptools import setup


__version__ = '0.1.0'


with open('README.md', encoding='utf-8') as f:
  _long_description = f.read()

setup(
    name='waymo-waymax',
    version=__version__,
    description='Waymo simulator for autonomous driving',
    long_description=_long_description,
    long_description_content_type='text/markdown',
    author='Waymax team',
    author_email='waymo-waymax@google.com',
    python_requires='>=3.10',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.20',
        'jax>=0.4.6',
        'tensorflow>=2.11.0',
        'chex>=0.1.6',
        'dm_env>=1.6',
        'flax>=0.6.7',
        'matplotlib>=3.7.1',
        'dm-tree>=0.1.8',
        'immutabledict>=2.2.3',
        'Pillow>=9.4.0',
        'mediapy>=1.1.6',
        'tqdm>=4.65.0',
        'absl-py>=1.4.0',
    ],
    url='https://github.com/waymo-research/waymax',
    license='Apache-2.0',
)
