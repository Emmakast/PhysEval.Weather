"""PhysMetrics.Weather: Physical consistency evaluation framework for AI Weather Prediction models.
"""

from physmetrics_weather.physics_metrics import (
    compute_conservation_scalars,
    compute_drift_percentages,
    compute_drift_slope,
    compute_dry_air_mass,
    compute_geostrophic_imbalance,
    compute_hydrostatic_imbalance,
    compute_ke_spectrum,
    compute_lapse_rate_wasserstein,
    compute_pure_tcwv,
    compute_q_spectrum,
    compute_spectral_scores,
    compute_total_energy,
    compute_water_mass,
    derive_surface_pressure,
    get_grid_cell_area,
)

__version__ = "0.1.0"

__all__ = [
    "get_grid_cell_area",
    "derive_surface_pressure",
    "compute_dry_air_mass",
    "compute_water_mass",
    "compute_total_energy",
    "compute_ke_spectrum",
    "compute_q_spectrum",
    "compute_spectral_scores",
    "compute_hydrostatic_imbalance",
    "compute_geostrophic_imbalance",
    "compute_lapse_rate_wasserstein",
    "compute_drift_slope",
    "compute_drift_percentages",
    "compute_conservation_scalars",
    "compute_pure_tcwv",
]
