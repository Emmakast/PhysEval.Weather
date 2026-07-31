# PhysMetrics.Weather (physmetrics-weather)

[![Documentation Status](https://readthedocs.org/projects/physmetricsweather/badge/?version=latest)](https://physmetricsweather.readthedocs.io/en/latest/?badge=latest)

An open-source, unified framework for evaluating the **physical consistency**, **spectral resolution**, and **atmospheric balance** of Machine Learning Weather Prediction (MLWP) models.

It computes diagnostic physical metrics against ERA5 or IFS HRES reference datasets from [WeatherBench 2](https://weatherbench2.readthedocs.io/), supporting both **deterministic** and **probabilistic / ensemble** weather models.

---

## Authors

* **Emma Kasteleyn**
* **Timo Maier** (`timo.maier@tum.de`)

---

## Key Metrics Evaluated
| Metric Category | Metric | What it measures |
|---|---|---|
| **Conservation & Stability** | **Dry Air Mass drift** | Is global dry-air mass conserved over time? (Exagrams, %/day) |
| | **Water Mass drift** | Is global total atmospheric water mass conserved? (kg, %/day) |
| | **Total Energy drift** | Is global total atmospheric energy conserved? (Joules, %/day) |
| **Spectral Skill** | **Effective Resolution ($L_{eff}$)** | Spatial scale (km) where model loses skill vs. reference |
| | **Spectral Divergence** | 1-Wasserstein distance between true and predicted spectra |
| | **Spectral Residual** | Log-RMSE difference between energy spectra |
| **Atmospheric Balance** | **Hydrostatic Balance** | Hypsometric balance error RMSE ($m^2/s^2$) between 500 & 850 hPa |
| | **Geostrophic Balance** | Area-weighted wind balance RMSE ($m/s$) at 500 hPa |
| | **Lapse Rate Wasserstein** | Lapse rate distribution distance across geographical bands |


---

## Installation

The package supports Python `>= 3.10` up to `3.14`. You can manage dependencies using [`uv`](https://docs.astral.sh/uv/) (Recommended) or standard `pip`.

### 1. Using `uv` (Recommended / Preferred Method)

[`uv`](https://docs.astral.sh/uv/) is the recommended, high-performance package and environment manager for this project:

```bash
# 1. Clone the repository
git clone https://github.com/Emmakast/PhysMetrics.Weather.git
cd PhysMetrics.Weather

# 2. Sync dependencies and create virtual environment
uv sync --extra dev --extra docs

# 3. Activate the virtual environment
source .venv/bin/activate   # Linux / macOS
# .venv\Scripts\activate    # Windows
```

> **Note on `uv` filesystem warnings**: If your working directory is on a different disk partition/mount than your home directory (where `uv` stores cached wheels), `uv` automatically falls back to full copies and may display `warning: Failed to hardlink files`. This is completely safe. To suppress the warning, run `export UV_LINK_MODE=copy` or use `uv sync --link-mode=copy`.

### 2. Using standard `pip` (Local Clone / Source)

> **Note**: `physmetrics-weather` uses `pyproject.toml` (PEP 517/518 standard). `pip` reads dependencies directly from `pyproject.toml`, so no `requirements.txt` file is needed.

```bash
# 1. Clone the repository
git clone https://github.com/Emmakast/PhysMetrics.Weather.git
cd PhysMetrics.Weather

# 2. Create and activate a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate   # Linux / macOS
# .venv\Scripts\activate    # Windows

# 3. Install the package in editable mode
pip install -e .

# Optional: Install with development & documentation tools
pip install -e ".[dev,docs]"
```

### 3. Direct `pip` Install from GitHub (Without Cloning)

```bash
pip install git+https://github.com/Emmakast/PhysMetrics.Weather.git
```

After installation, the CLI commands `physmetrics-run` and `physmetrics-plot` are available on your system `PATH`:

| Command | Description |
|---|---|
| `physmetrics-run` | Stream WeatherBench 2 Zarr data, run physics metrics, output long-format CSV |
| `physmetrics-plot` | Generate publication-ready figures and visualization plots from output CSVs |

---

## Probabilistic & Ensemble Model Support

`physmetrics-weather` automatically detects extra ensemble dimensions in your datasets (e.g. `ens`, `realization`, `member`, `ensemble`, `number`).

* **Per-Member Evaluation**: Metrics are computed for each individual ensemble member.
* **Output Format**: The resulting long-format CSV includes an `ensemble_member` column (`0` for deterministic models, or member ID `0, 1, 2, ...` for ensemble realizations).

---

## Data Inputs & CLI Usage

> [!NOTE]
> **Execution Time Notice**: Evaluating a full 365-day year across multiple forecast lead horizons involves streaming and calculating spherical harmonic spectra and 3D integrals for over 1,000 data slices. A full-year run can take several hours depending on network bandwidth and CPU cores. For quick testing and examples, use `--dates 2020-01-01 2020-01-02`.

### 1. WeatherBench 2 Streaming (No Download Required)

Stream data directly from public WeatherBench 2 Google Cloud Storage buckets:

```bash
# Evaluate Pangu-Weather predictions against ERA5 for specific dates
physmetrics-run \
  --model pangu \
  --prediction-zarr gs://weatherbench2/datasets/pangu/2018-2022_0012_0p25.zarr \
  --ref-zarr gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr \
  --dates 2020-01-01 2020-01-02 \
  --workers 4 \
  --output-dir ./results
```

### 2. Custom Zarr Dataset Evaluation

Point `--prediction-zarr` at any local or cloud Zarr store:

```bash
physmetrics-run \
  --model my_model \
  --prediction-zarr /path/to/my_model_forecasts.zarr \
  --ref-zarr /path/to/my_model_reference.zarr \
  --dates 2020-01-01 2020-01-02 \
  --output-dir ./results
```

Required variable names (or standard aliases):
* **Surface Pressure**: `surface_pressure`, `sp`, `ps`
* **Temperature**: `temperature`, `t`
* **Zonal Wind**: `u_component_of_wind`, `u`
* **Meridional Wind**: `v_component_of_wind`, `v`
* **Specific Humidity**: `specific_humidity`, `q`
* **Geopotential**: `geopotential`, `z`

---

## CLI Options

### `physmetrics-run`

| Option | Default | Description |
|---|---|---|
| `--year` | `2022` | Year to evaluate |
| `--dates` | — | Specific ISO dates, e.g. `2020-01-01 2020-01-02` |
| `--month` | — | Specific month, e.g. `2020-01` |
| `--model` | `model` | Model name identifier |
| `--prediction-zarr` | WB2 path | Zarr store URL or local path for model predictions |
| `--ref-zarr` | ERA5 WB2 path | Zarr store URL or local path for reference dataset |
| `--lead-times` | `12h,5d,10d` | Comma-separated forecast lead times |
| `--workers` | `4` | Parallel worker process count |
| `--output-dir` | `./results` | Directory for output CSV results |
| `--output` | — | Custom destination CSV file path |
| `--mode` | `joint` | Evaluation mode (`joint`, `ref`, `prediction`, `model`) |
| `--spectra` | `KE:500` | Comma-separated target spectra specs, e.g. `KE:500,Q:500,T:850` |
| `--extended-spectra` | off | Compute additional Q and 850 hPa spectra |
| `--quiet` | off | Suppress verbose progress logging |

### `physmetrics-plot`

```bash
physmetrics-plot --results-dir ./results --outdir ./plots
```

| Option | Default | Description |
|---|---|---|
| `--results-dir` | `./results` | Path to directory containing output CSV files |
| `--outdir` | `./plots` | Path to directory for saving generated figures |
| `--reference-label` | `auto` | Reference dataset label override (`auto`, `ERA5`, `IFS`) |

---

## Python API Usage

### Object-Oriented Evaluation Pipeline

Run evaluation workflows programmatically using `EvaluationConfig` and `EvaluationPipeline`:

```python
from pathlib import Path
from physmetrics_weather import EvaluationConfig, EvaluationPipeline

config = EvaluationConfig(
    dates=["2020-01-01", "2020-01-02"],
    prediction_zarr="gs://weatherbench2/datasets/aurora/2022-1440x721.zarr",
    model_name="aurora",
    output_csv=Path("./results/physics_evaluation_aurora_2020.csv"),
    workers=4,
)

pipeline = EvaluationPipeline(config)
df = pipeline.run()
```

### Object-Oriented Plotter

Render diagnostic plots programmatically using `PlotterConfig` and `PhysicsPlotter`:

```python
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
```

---

## Output CSV Formats

`physmetrics-run` generates four types of CSV files in the output directory:

### 1. `physics_evaluation_*.csv` (Main summary metrics)
```csv
date,lead_time_hours,metric_name,model_value,ref_value,n_levels,sp_method,ensemble_member
2020-01-01,12,hydrostatic_rmse,1.23,0.98,13,direct_sp,0
2020-01-01,12,geostrophic_rmse,2.45,2.10,13,direct_sp,0
2020-01-01,120,dry_mass_drift_pct_per_day,-0.002,,13,direct_sp,0
```

### 2. `time_series_*.csv` (Conservation and balance tracking)
```csv
date,forecast_hour,dry_mass_Eg,water_mass_kg,total_energy_J,hydrostatic_rmse,geostrophic_rmse,sp_method,ensemble_member
2020-01-01,12.0,5.12e18,1.2e16,4.5e21,1.23,2.45,direct_sp,0
```

### 3. `spectra_*.csv` (Spherical harmonic spectra power)
```csv
date,lead_hours,variable,wavenumber,power_pred,power_ref,ensemble_member
2020-01-01,12,KE:500,10,450.2,455.1,0
```

### 4. `lapse_rate_dist_*.csv` (Lapse rate probability distributions)
```csv
date,lead_hours,region,bin_edge_lower,freq_pred,freq_ref,ensemble_member
2020-01-01,12,tropics,-10.5,0.012,0.014,0
```

---

## Documentation

Full Sphinx documentation is available in the `docs/` directory.

### Build and View Locally

```bash
# 1. Install documentation dependencies (if not already installed)
pip install -e ".[docs]"

# 2. Build Sphinx HTML documentation
sphinx-build -b html docs docs/_build/html
# (Or using uv: uv run sphinx-build -b html docs docs/_build/html)

# 3. Serve locally at http://localhost:8000
python3 -m http.server 8000 --directory docs/_build/html
```

---

## Testing & Verification

Run the comprehensive unit test suite:

```bash
# Install development dependencies (if not already installed)
pip install -e ".[dev]"

# Run pytest test suite
pytest
# (Or using uv: uv run pytest)

# Verify wheel and source distribution build
uv build
```

---

## Citation

If you use **PhysMetrics.Weather** in your research, please cite our paper:

```bibtex
@misc{kasteleyn2026physmetricsweatherevaluationframeworkphysical,
      title={PhysMetrics.Weather: An Evaluation Framework for Physical Consistency in ML Weather Models}, 
      author={Emma Kasteleyn and Timo Maier and Axel Lauer and Veronika Eyring and Pierre Gentine and Ana Lucic},
      year={2026},
      eprint={2606.10642},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2606.10642}, 
}
```

---

## License

This project is licensed under the [MIT License](LICENCE).

