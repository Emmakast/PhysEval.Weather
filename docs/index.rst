PhysMetrics.Weather Documentation
======================================

**PhysMetrics.Weather** is a comprehensive Python framework designed to evaluate the physical consistency, spectral resolution, and conservation properties of Machine Learning Weather Prediction (MLWP) models.

Key Features
------------

* **Mass, Water & Energy Conservation**: Calculate global dry air mass (Eg), water mass (kg), and total atmospheric energy (J) drift rates.
* **Spectral Analysis & Effective Resolution**: Compute spherical harmonic kinetic energy (KE) or specific humidity (Q) spectra, spectral divergence (1-Wasserstein), spectral residual (log-RMSE), and effective spatial resolution (:math:`L_{eff}`).
* **Atmospheric Balance Metrics**: Evaluate hydrostatic hypsometric balance RMSE, geostrophic wind balance RMSE across pressure levels and compute lapse rate distributions (area-weighted Wasserstein).
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

Citation
--------

If you use **PhysMetrics.Weather** in your research, please cite our paper:

.. code-block:: bibtex

   @misc{kasteleyn2026physmetricsweatherevaluationframeworkphysical,
         title={PhysMetrics.Weather: An Evaluation Framework for Physical Consistency in ML Weather Models}, 
         author={Emma Kasteleyn and Timo Maier and Axel Lauer and Veronika Eyring and Pierre Gentine and Ana Lucic},
         year={2026},
         eprint={2606.10642},
         archivePrefix={arXiv},
         primaryClass={cs.LG},
         url={https://arxiv.org/abs/2606.10642}, 
   }

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
