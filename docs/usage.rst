Usage Guide
===========

``physmetrics-weather`` provides two command-line interface (CLI) utilities: ``physmetrics-run`` for evaluation streaming and metric calculation, and ``physmetrics-plot`` for visualization.

Running Metric Evaluation (physmetrics-run)
-------------------------------------------

The ``physmetrics-run`` command streams model predictions and reference datasets, evaluating all physical metrics.

Command Syntax
~~~~~~~~~~~~~~

.. code-block:: bash

   physmetrics-run [OPTIONS]

Key Options
~~~~~~~~~~~

* ``--year INTEGER``: Year to evaluate (default: 2022).
* ``--dates STRING [STRING ...]``: Specific ISO dates to evaluate (e.g. ``2022-01-01 2022-01-15``).
* ``--month STRING``: Evaluate all days of a month (e.g. ``2022-01``).
* ``--prediction-zarr STRING``: Path or GCS URL to model prediction Zarr dataset.
* ``--ref-zarr STRING``: Path or GCS URL to reference ground truth Zarr dataset.
* ``--lead-times STRING``: Comma-separated list of target lead times (default: ``12h,5d,10d``).
* ``--workers INTEGER``: Number of parallel worker processes (default: 4).
* ``--output-dir PATH``: Output directory path for generated CSV files (default: ``./results``).
* ``--output PATH``: Custom destination CSV file path.
* ``--mode {joint,ref,prediction,model}``: Evaluation mode (default: ``joint``).

Example Commands
~~~~~~~~~~~~~~~~

Evaluate single model over specific dates:

.. code-block:: bash

   physmetrics-run --dates 2022-01-01 2022-01-15 --model aurora --workers 4

Evaluate custom Zarr prediction dataset:

.. code-block:: bash

   physmetrics-run \
     --prediction-zarr gs://weatherbench2/datasets/aurora/2022-1440x721.zarr \
     --model aurora \
     --output-dir ./results

Generating Visualizations (physmetrics-plot)
--------------------------------------------

The ``physmetrics-plot`` command reads the generated long-format CSV files and renders publication-ready plots.

Command Syntax
~~~~~~~~~~~~~~

.. code-block:: bash

   physmetrics-plot --results-dir ./results --outdir ./plots

Key Options
~~~~~~~~~~~

* ``--results-dir PATH``: Path to directory containing output CSV files (default: ``./results``).
* ``--outdir PATH``: Destination directory for generated plots (default: ``./plots``).
* ``--reference-label {ERA5,IFS}``: Reference dataset label override for plot legends.

Generated Figures
~~~~~~~~~~~~~~~~~

* ``ts_dry_mass_Eg.png``: Dry air mass relative drift timeseries.
* ``ts_water_mass_kg.png``: Atmospheric water mass relative drift timeseries.
* ``ts_total_energy_J.png``: Total atmospheric energy relative drift timeseries.
* ``ts_hydrostatic_rmse.png``: Hydrostatic balance RMSE timeseries.
* ``ts_geostrophic_rmse.png``: Geostrophic balance RMSE timeseries.
* ``spectra_ke_12h.png``, ``spectra_ke_120h.png``, ``spectra_ke_240h.png``: Kinetic energy spectra at lead times.
