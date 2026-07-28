"""PhysMetrics.Weather: Physical consistency evaluation framework for AI Weather Prediction models.
"""

from physmetrics_weather.physics_metrics import (
    DatasetValidator,
    MetricResult,
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
    compute_scalar_spectrum,
    compute_spectral_scores,
    compute_total_energy,
    compute_water_mass,
    derive_surface_pressure,
    get_grid_cell_area,
)
from physmetrics_weather.plot_neurips_metrics import (
    PhysicsPlotter,
    PlotterConfig,
    generate_all_plots,
)
from physmetrics_weather.run_all_metrics import (
    EvaluationConfig,
    EvaluationPipeline,
    run_evaluation,
)

__version__ = "0.1.0"

__all__ = [
    # Metrics Core
    "MetricResult",
    "DatasetValidator",
    "get_grid_cell_area",
    "derive_surface_pressure",
    "compute_dry_air_mass",
    "compute_water_mass",
    "compute_total_energy",
    "compute_ke_spectrum",
    "compute_q_spectrum",
    "compute_scalar_spectrum",
    "compute_spectral_scores",
    "compute_hydrostatic_imbalance",
    "compute_geostrophic_imbalance",
    "compute_lapse_rate_wasserstein",
    "compute_drift_slope",
    "compute_drift_percentages",
    "compute_conservation_scalars",
    "compute_pure_tcwv",
    # Evaluation Pipeline
    "EvaluationConfig",
    "EvaluationPipeline",
    "run_evaluation",
    # Plotting & Visualization
    "PlotterConfig",
    "PhysicsPlotter",
    "generate_all_plots",
]
