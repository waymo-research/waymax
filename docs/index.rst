Waymax: An accelerated simulator for autonomous driving research
================================================================

Waymax is a lightweight, multi-agent, `JAX`_-based simulator for autonomous driving research based on the Waymo Open Motion Dataset.

.. grid:: 2
   :margin: 0
   :padding: 0
   :gutter: 2

   .. grid-item-card:: :material-regular:`rocket_launch;2em` Getting Started
      :link: getting-started
      :link-type: ref
      :class-card: sd-border-0

   .. grid-item-card:: :material-regular:`library_books;2em` API Documentation
      :link: autoapi/waymax/index
      :link-type: doc
      :class-card: sd-border-0


Installation
------------
.. code-block:: bash

  pip install git+https://github.com/waymo-research/waymax.git@main#egg=waymo-waymax

For additional installation options, see the `Install Guide`_ in the project README.


.. toctree::
   :hidden:
   :maxdepth: 1

   getting_started


.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: API Documentation

   autoapi/waymax/agents/index
   autoapi/waymax/dataloader/index
   autoapi/waymax/datatypes/index
   autoapi/waymax/env/index
   autoapi/waymax/metrics/index
   autoapi/waymax/rewards/index
   autoapi/waymax/utils/index
   autoapi/waymax/visualization/index
   autoapi/waymax/config/index

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Further Resources

   genindex
   modindex

.. _JAX: https://github.com/jax-ml/jax
.. _Install Guide: https://github.com/waymo-research/waymax#installation
