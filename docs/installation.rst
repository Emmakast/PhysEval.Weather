Installation Guide
==================

This guide describes how to install ``physmetrics-weather`` using ``pip``, ``uv``, or directly from source.

Requirements
------------

* **Python**: ``>= 3.10`` (supports up to Python ``3.14``)
* **Key Dependencies**: ``numpy``, ``pandas``, ``xarray``, ``dask``, ``scipy``, ``matplotlib``, ``seaborn``, ``zarr``, ``gcsfs``, ``pyshtools``

Installing via PyPI (pip)
-------------------------

Once published on PyPI, install the latest release using ``pip``:

.. code-block:: bash

   pip install physmetrics-weather

To install with development and documentation tools:

.. code-block:: bash

   pip install "physmetrics-weather[dev,docs]"

Installing via uv
-----------------

`uv <https://github.com/astral-sh/uv>`_ is a fast Python package installer and resolution tool.

To install ``physmetrics-weather`` into your virtual environment with ``uv``:

.. code-block:: bash

   uv pip install physmetrics-weather

Or add it to your project dependencies:

.. code-block:: bash

   uv add physmetrics-weather

Installing from Source
----------------------

Clone the repository and install the package in editable mode:

.. code-block:: bash

   git clone https://github.com/Emmakast/PhysMetrics.Weather.git
   cd PhysMetrics.Weather
   pip install -e .

Using ``uv`` for local development:

.. code-block:: bash

   uv sync --extra dev --extra docs
   uv run pytest
