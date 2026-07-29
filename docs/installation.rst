Installation Guide
==================

This guide describes how to install ``physmetrics-weather`` using standard ``pip`` or ``uv``.

.. note::
   ``physmetrics-weather`` uses modern ``pyproject.toml`` configuration (PEP 517/518 standard). ``pip`` automatically reads dependencies directly from ``pyproject.toml`` without requiring a ``requirements.txt`` file.

Requirements
------------

* **Python**: ``>= 3.10`` (supports up to Python ``3.14``)
* **Key Dependencies**: ``numpy``, ``pandas``, ``xarray``, ``dask``, ``scipy``, ``matplotlib``, ``seaborn``, ``zarr``, ``gcsfs``, ``pyshtools``

Option 1: Using uv (Recommended / Preferred Method)
---------------------------------------------------

`uv <https://github.com/astral-sh/uv>`_ is the recommended, high-performance Python package installer and resolution tool.

1. Clone the repository and sync environment dependencies:

.. code-block:: bash

   git clone https://github.com/Emmakast/PhysMetrics.Weather.git
   cd PhysMetrics.Weather
   uv sync --extra dev --extra docs

2. Activate the virtual environment:

.. code-block:: bash

   source .venv/bin/activate   # Linux / macOS
   # .venv\Scripts\activate    # Windows

Option 2: Installing via Standard pip (Local Clone / Source)
------------------------------------------------------------

1. Clone the repository:

.. code-block:: bash

   git clone https://github.com/Emmakast/PhysMetrics.Weather.git
   cd PhysMetrics.Weather

2. Create and activate a virtual environment (recommended):

.. code-block:: bash

   python3 -m venv .venv
   source .venv/bin/activate   # Linux / macOS
   # .venv\Scripts\activate    # Windows

3. Install the package in editable mode:

.. code-block:: bash

   pip install -e .

To install optional development and documentation dependencies:

.. code-block:: bash

   pip install -e ".[dev,docs]"

Option 3: Installing Directly from GitHub via pip
-------------------------------------------------

If you do not need a local git checkout, install directly using ``pip``:

.. code-block:: bash

   pip install git+https://github.com/Emmakast/PhysMetrics.Weather.git

Option 4: Installing via PyPI (pip)
-----------------------------------

Once published on PyPI, install the latest release using ``pip``:

.. code-block:: bash

   pip install physmetrics-weather

