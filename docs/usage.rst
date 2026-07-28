Usage Guide
===========

``physmetrics-weather`` provides both command-line interface (CLI) utilities and a Python object-oriented API for metric calculation and visualization.

1. Command-Line Interface (CLI)
-------------------------------

Running Metric Evaluation (physmetrics-run)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``physmetrics-run`` command streams model predictions and reference datasets, evaluating all physical metrics.

.. code-block:: bash

   physmetrics-run [OPTIONS]

Key CLI Options:

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

CLI Example:

.. code-block:: bash

   physmetrics-run \
     --prediction-zarr gs://weatherbench2/datasets/aurora/2022-1440x721.zarr \
     --model aurora \
     --dates 2022-01-01 2022-01-15 \
     --output-dir ./results

Generating Visualizations (physmetrics-plot)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``physmetrics-plot`` command reads long-format CSV files and renders publication-ready plots.

.. code-block:: bash

   physmetrics-plot --results-dir ./results --outdir ./plots

2. Python API Usage
-------------------

Running Evaluation Pipelines
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``EvaluationConfig`` and ``EvaluationPipeline`` to run evaluation workflows programmatically:

.. code-block:: python

   from pathlib import Path
   from physmetrics_weather import EvaluationConfig, EvaluationPipeline

   # 1. Define configuration settings
   config = EvaluationConfig(
       dates=["2022-01-01", "2022-01-02"],
       prediction_zarr="gs://weatherbench2/datasets/aurora/2022-1440x721.zarr",
       model_name="aurora",
       output_csv=Path("./results/physics_evaluation_aurora_2022.csv"),
       workers=4,
   )

   # 2. Execute pipeline
   pipeline = EvaluationPipeline(config)
   df = pipeline.run()

   print(f"Evaluated {len(df)} metric records.")

Rendering Diagnostic Figures
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``PlotterConfig`` and ``PhysicsPlotter`` to generate figure plots programmatically:

.. code-block:: python

   from pathlib import Path
   from physmetrics_weather import PlotterConfig, PhysicsPlotter

   # 1. Configure plotter
   config = PlotterConfig(
       results_dir=Path("./results"),
       outdir=Path("./plots"),
       style="whitegrid",
       dpi=300,
       file_format="png",
   )

   # 2. Render plots
   plotter = PhysicsPlotter(config)
   plot_paths = plotter.generate_all()

   for path in plot_paths:
       print(f"Saved plot: {path}")

Generated Figures
-----------------

* ``ts_dry_mass_Eg.png``: Dry air mass relative drift timeseries.
* ``ts_water_mass_kg.png``: Atmospheric water mass relative drift timeseries.
* ``ts_total_energy_J.png``: Total atmospheric energy relative drift timeseries.
* ``ts_hydrostatic_rmse.png``: Hydrostatic balance RMSE timeseries.
* ``ts_geostrophic_rmse.png``: Geostrophic balance RMSE timeseries.
* ``spectra_ke_*.png``: Kinetic energy spectra at lead times.
