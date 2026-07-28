Examples
========

This page provides practical code examples for using ``physmetrics_weather`` via Python API and CLI.

Evaluating Single Slices via Python API
---------------------------------------

.. code-block:: python

   import xarray as xr
   from physmetrics_weather import (
       get_grid_cell_area,
       derive_surface_pressure,
       compute_dry_air_mass,
       compute_water_mass,
       compute_ke_spectrum,
   )

   # Load forecast snapshot from WeatherBench 2
   ds_model = xr.open_zarr("gs://weatherbench2/datasets/aurora/2022-1440x721.zarr", storage_options={"token": "anon"})
   ds_ref = xr.open_zarr("gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr", storage_options={"token": "anon"})

   snapshot = ds_model.sel(time="2022-01-01T00:00:00").isel(prediction_timedelta=2)
   area = get_grid_cell_area(snapshot)
   ps = derive_surface_pressure(snapshot, ds_ref)

   # Compute conservation metrics
   dry_mass_eg = compute_dry_air_mass(snapshot, ps, area)
   water_mass_kg = compute_water_mass(snapshot, ps, area)

   print(f"Global Dry Air Mass: {dry_mass_eg:.6f} Eg")
   print(f"Global Water Mass: {water_mass_kg:.6e} kg")

Batch Evaluation via EvaluationPipeline
---------------------------------------

Use ``EvaluationPipeline`` to evaluate weather model predictions across multiple dates:

.. code-block:: python

   from pathlib import Path
   from physmetrics_weather import EvaluationConfig, EvaluationPipeline

   config = EvaluationConfig(
       dates=["2022-01-01", "2022-01-02", "2022-01-03"],
       prediction_zarr="gs://weatherbench2/datasets/pangu/2018-2022_0012_0p25.zarr",
       model_name="pangu",
       output_csv=Path("./results/physics_evaluation_pangu_2022.csv"),
       workers=4,
   )

   pipeline = EvaluationPipeline(config)
   df = pipeline.run()

Plot Generation via PhysicsPlotter
----------------------------------

Render diagnostic plots from output summary CSVs using ``PhysicsPlotter``:

.. code-block:: python

   from pathlib import Path
   from physmetrics_weather import PlotterConfig, PhysicsPlotter

   config = PlotterConfig(
       results_dir=Path("./results"),
       outdir=Path("./plots"),
       style="whitegrid",
       dpi=300,
   )

   plotter = PhysicsPlotter(config)
   plots = plotter.generate_all()

Probabilistic / Ensemble Model Evaluation
------------------------------------------

When a dataset contains an ensemble dimension (e.g. ``"ens"`` or ``"member"``), metrics are evaluated per ensemble member:

.. code-block:: python

   from physmetrics_weather import compute_dry_air_mass, get_grid_cell_area

   # ds_ensemble has dimension 'ens' with multiple members
   area = get_grid_cell_area(ds_ensemble)
   ps = ds_ensemble["surface_pressure"]

   # Returns dictionary mapping member ID -> dry air mass (Eg)
   dry_mass_members = compute_dry_air_mass(ds_ensemble, ps, area)

   for member_id, dry_mass in dry_mass_members.items():
       print(f"Member {member_id}: {dry_mass:.6f} Eg")

CLI Batch Running
-----------------

.. code-block:: bash

   physmetrics-run \
     --prediction-zarr /path/to/model_forecast.zarr \
     --model my_model \
     --dates 2022-01-01 \
     --output-dir ./results
