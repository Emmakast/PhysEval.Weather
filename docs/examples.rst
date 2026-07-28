Examples
========

This page provides practical examples of using ``physmetrics_weather`` via Python API and CLI for deterministic and probabilistic/ensemble models using WeatherBench 2 data.

Evaluating Deterministic Models via Python API
----------------------------------------------

.. code-block:: python

   import xarray as xr
   from physmetrics_weather import (
       get_grid_cell_area,
       derive_surface_pressure,
       compute_dry_air_mass,
       compute_water_mass,
       compute_total_energy,
       compute_ke_spectrum,
   )

   # Load forecast slice from WeatherBench 2
   ds_model = xr.open_zarr("gs://weatherbench2/datasets/aurora/2022-1440x721.zarr", storage_options={"token": "anon"})
   ds_ref = xr.open_zarr("gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr", storage_options={"token": "anon"})

   # Select forecast slice
   snapshot = ds_model.sel(time="2022-01-01T00:00:00").isel(prediction_timedelta=2) # 12h forecast

   # Compute grid cell area
   area = get_grid_cell_area(snapshot)

   # Derive surface pressure if missing
   ps = derive_surface_pressure(snapshot, ds_ref)

   # Compute mass and energy metrics
   dry_mass_eg = compute_dry_air_mass(snapshot, ps, area)
   water_mass_kg = compute_water_mass(snapshot, ps, area)

   print(f"Global Dry Air Mass: {dry_mass_eg:.6f} Eg")
   print(f"Global Water Mass: {water_mass_kg:.6e} kg")

   # Compute 500 hPa KE spectrum
   wavenumber, ke_energy = compute_ke_spectrum(snapshot, level=500.0)

Evaluating Probabilistic / Ensemble Models
-------------------------------------------

When a dataset contains an ensemble dimension (e.g., ``"ens"``, ``"realization"``, or ``"member"``), ``physmetrics_weather`` automatically evaluates metrics across each ensemble member.

.. code-block:: python

   import xarray as xr
   from physmetrics_weather import compute_dry_air_mass, get_grid_cell_area

   # Mock ensemble dataset with dimension 'ens'
   # ds_ensemble has shape (ens: 10, level: 5, latitude: 32, longitude: 64)

   area = get_grid_cell_area(ds_ensemble)
   ps = ds_ensemble["surface_pressure"]

   # Returns a dictionary mapping member ID -> dry air mass (Eg)
   dry_mass_members = compute_dry_air_mass(ds_ensemble, ps, area)

   for member_id, dry_mass in dry_mass_members.items():
       print(f"Ensemble Member {member_id}: {dry_mass:.6f} Eg")

CLI Ensemble Batch Running
--------------------------

To run an ensemble dataset evaluation via CLI:

.. code-block:: bash

   physmetrics-run \
     --prediction-zarr /path/to/ensemble_forecast.zarr \
     --model my_ensemble_model \
     --dates 2022-01-01 \
     --output-dir ./results

The generated CSV will contain an ``ensemble_member`` column detailing per-member metrics:

.. code-block:: text

   date,lead_time_hours,metric_name,model_value,ref_value,n_levels,sp_method,ensemble_member
   2022-01-01,12,dry_mass_drift_pct_per_day,0.0012,,5,direct_sp,0
   2022-01-01,12,dry_mass_drift_pct_per_day,0.0015,,5,direct_sp,1
   2022-01-01,12,dry_mass_drift_pct_per_day,0.0011,,5,direct_sp,2
