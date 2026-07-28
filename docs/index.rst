PhysMetrics.Weather Documentation
======================================

**PhysMetrics.Weather** is a comprehensive Python framework designed to evaluate the physical consistency, spectral resolution, and conservation properties of Machine Learning Weather Prediction (MLWP) models.

Key Features
------------

* **Mass, Water & Energy Conservation**: Calculate global dry air mass (Eg), water mass (kg), and total atmospheric energy (J) drift rates.
* **Spectral Analysis & Effective Resolution**: Compute spherical harmonic kinetic energy (KE) and specific humidity (Q) spectra, spectral divergence (1-Wasserstein), and effective spatial resolution (:math:`L_{eff}`).
* **Atmospheric Balance Metrics**: Evaluate hydrostatic hypsometric balance RMSE and geostrophic wind balance RMSE across pressure levels.
* **Thermal Structure**: Compute environmental lapse rate distributions and area-weighted Wasserstein distance across geographical regions.
* **Probabilistic & Ensemble Support**: Seamlessly detect and process extra ensemble dimensions (e.g. ``ens``, ``realization``, ``member``), producing long-format outputs per member for downstream aggregation.
* **WeatherBench 2 Integration**: Direct Zarr streaming from public WeatherBench 2 Google Cloud Storage buckets.

Contents
--------

.. toctree::
   :maxdepth: 2

   installation
   usage
   examples
   api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
